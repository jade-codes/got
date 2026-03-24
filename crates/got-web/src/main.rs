// ---------------------------------------------------------------------------
// got-web: conversational incoherence visualiser.
//
// A self-contained axum server that:
//   1. Serves a single-page D3.js frontend at GET /
//   2. Analyses conversation coherence at POST /api/conversation/analyse
//   3. Returns a demo conversation at GET /api/demo-conversation
//
// Loads a reference model's unembedding matrix (.gotue) and vocabulary,
// then resolves value terms to embeddings in the model's hidden space.
// Value terms come from a taxonomy TOML file (--values) or fall back to
// a built-in default list.
// ---------------------------------------------------------------------------

mod api;

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use got_core::geometry::CausalGeometry;
use got_core::UnembeddingMatrix;
use got_incoherence::coherence::CoherenceConfig;
use got_incoherence::embeddings::{EmbeddingSource, PrecomputedEmbeddings, UnembeddingLookup};
use got_web::AppState;
use got_web::proxy_api::ProxyState;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "got-web", about = "Conversational incoherence visualiser")]
struct Args {
    /// Path to .gotue unembedding matrix file.
    #[arg(long)]
    geometry: String,

    /// Path to vocabulary JSON array.
    #[arg(long)]
    vocab: String,

    /// Path to value taxonomy TOML file.
    /// When provided, value terms and descriptions are loaded from this file
    /// and descriptions are embedded by averaging token vectors from the
    /// reference model's unembedding matrix.
    /// When omitted, falls back to built-in single-token value terms.
    #[arg(long)]
    values: Option<String>,

    /// Path to demo conversation JSON with matching embedding dimensions.
    /// Defaults to data/models/gpt2-demo-conversation.json.
    #[arg(long)]
    demo_conversation: Option<String>,

    /// Listen address (default: 127.0.0.1:3000).
    #[arg(long, default_value = "127.0.0.1:3000")]
    listen: String,

    /// Path to static files directory (default: auto-detected relative to binary).
    #[arg(long)]
    static_dir: Option<String>,
}

// ---------------------------------------------------------------------------
// Value taxonomy
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct ValueTaxonomy {
    values: Vec<ValueEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ValueEntry {
    name: String,
    description: String,
    #[allow(dead_code)]
    #[serde(default)]
    cluster: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    antonyms: Vec<String>,
}

/// Embed a description by averaging token embeddings from the reference model.
///
/// Tokenizes by whitespace, looks up each token in the unembedding matrix,
/// and averages the found vectors.  This produces a multi-token concept
/// vector in the reference model's own hidden space.
fn embed_description(description: &str, lookup: &UnembeddingLookup) -> Option<Vec<f32>> {
    let dim = lookup.hidden_dim();
    let mut sum = vec![0.0f32; dim];
    let mut matched = 0usize;

    for token in description.split_whitespace() {
        let clean: String = token
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
            .collect();
        if clean.is_empty() {
            continue;
        }
        if let Some(emb) = lookup.embed(&clean) {
            for (s, e) in sum.iter_mut().zip(emb.iter()) {
                *s += e;
            }
            matched += 1;
        }
    }

    if matched == 0 {
        return None;
    }

    let scale = 1.0 / matched as f32;
    for s in sum.iter_mut() {
        *s *= scale;
    }
    Some(sum)
}

// Fallback value terms when no taxonomy file is provided.
const DEFAULT_VALUE_TERMS: &[&str] = &[
    "compassion", "courage", "cowardice", "cruelty",
    "fairness", "freedom", "honesty", "innovation",
    "oppression", "secrecy", "tradition", "transparency", "wisdom",
];

/// Return the demo conversation (pre-built scenario with message embeddings).
async fn demo_conversation(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        state.demo_conversation_json.clone(),
    )
}

// ---------------------------------------------------------------------------
// State builder
// ---------------------------------------------------------------------------

/// Load a .gotue binary file into an UnembeddingMatrix.
fn load_gotue(path: &str) -> UnembeddingMatrix {
    let data = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));

    if data.len() < 14 || &data[0..4] != b"GOTU" {
        panic!("{path}: not a valid .gotue file (bad magic)");
    }

    let mut offset = 4;
    // version u16 LE
    let _version = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
    offset += 2;
    // vocab_size u32 LE
    let vocab_size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    // hidden_dim u32 LE
    let hidden_dim = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let total = vocab_size * hidden_dim;
    let total_bytes = total * 4;
    if offset + total_bytes > data.len() {
        panic!(
            "{path}: truncated — need {} bytes from offset {}, file has {}",
            total_bytes, offset, data.len()
        );
    }

    let mut values = Vec::with_capacity(total);
    let float_slice = &data[offset..offset + total_bytes];
    for chunk in float_slice.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }

    UnembeddingMatrix::new(vocab_size, hidden_dim, values)
        .expect("unembedding data length mismatch")
}

/// Build application state from the reference model and value terms.
fn build_state(args: &Args) -> AppState {
    eprintln!("Loading unembedding matrix from {}...", args.geometry);
    let matrix = load_gotue(&args.geometry);
    eprintln!(
        "  {} vocab × {} hidden dim",
        matrix.vocab_size, matrix.hidden_dim
    );

    let hidden_dim = matrix.hidden_dim;

    eprintln!("Loading vocabulary from {}...", args.vocab);
    let vocab_json = std::fs::read_to_string(&args.vocab)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.vocab));
    let vocab_raw: Vec<String> = serde_json::from_str(&vocab_json)
        .unwrap_or_else(|e| panic!("failed to parse vocab JSON: {e}"));

    // Strip BPE prefix (Ġ) so terms like "honesty" match "Ġhonesty"
    let vocab_clean: Vec<String> = vocab_raw
        .iter()
        .map(|t| t.replace('Ġ', ""))
        .collect();

    // Build lookup with cleaned vocab
    let lookup = UnembeddingLookup::new(vocab_clean, matrix)
        .expect("vocab/matrix mismatch");

    // Resolve value terms to embeddings in the reference model's hidden space.
    let mut term_embeddings = HashMap::new();
    let mut available_terms = Vec::new();

    if let Some(ref values_path) = args.values {
        // Taxonomy mode: embed descriptions by averaging token vectors.
        eprintln!("Loading value taxonomy from {values_path}...");
        let toml_str = std::fs::read_to_string(values_path)
            .unwrap_or_else(|e| panic!("failed to read {values_path}: {e}"));
        let taxonomy: ValueTaxonomy = toml::from_str(&toml_str)
            .unwrap_or_else(|e| panic!("failed to parse taxonomy TOML: {e}"));
        eprintln!("  {} value entries loaded", taxonomy.values.len());

        for entry in &taxonomy.values {
            if let Some(emb) = embed_description(&entry.description, &lookup) {
                let tokens_in_desc = entry.description.split_whitespace().count();
                let matched = entry.description
                    .split_whitespace()
                    .filter(|t| {
                        let clean: String = t.to_lowercase()
                            .chars()
                            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                            .collect();
                        !clean.is_empty() && lookup.embed(&clean).is_some()
                    })
                    .count();
                eprintln!(
                    "  '{}': embedded description ({}/{} tokens matched)",
                    entry.name, matched, tokens_in_desc
                );
                term_embeddings.insert(entry.name.clone(), emb);
                available_terms.push(entry.name.clone());
            } else {
                eprintln!("  warning: '{}' — no tokens matched in vocabulary (skipped)", entry.name);
            }
        }
    } else {
        // Fallback: single-token lookup from built-in list.
        eprintln!("No --values file; using built-in single-token value terms.");
        for &term in DEFAULT_VALUE_TERMS {
            if let Some(emb) = lookup.embed(term) {
                term_embeddings.insert(term.to_string(), emb);
                available_terms.push(term.to_string());
            } else {
                eprintln!("  warning: term '{term}' not in vocabulary (skipped)");
            }
        }
    }

    available_terms.sort();
    eprintln!("  resolved {} value terms", available_terms.len());

    if term_embeddings.is_empty() {
        panic!("no value terms could be resolved — check taxonomy or vocabulary");
    }

    // Mean-centre term embeddings for pairwise analysis.
    //
    // GPT-2's value terms all live in the same positive quadrant (raw cosines
    // 0.24–0.76).  Mean-centring removes the dominant shared component and
    // exposes contrastive structure: centred cosines range [-0.23, +0.53],
    // with compassion↔secrecy, cruelty↔integrity surfacing as opposed and
    // bravery↔courage, creativity↔innovation as aligned.
    //
    // Detection (z-scored logits) uses the RAW embeddings — centering
    // subtracts a constant from all logits, which z-scoring removes anyway.
    eprintln!("Mean-centring term embeddings for pairwise analysis...");
    let n_terms = term_embeddings.len();
    let mut mean_vec = vec![0.0f32; hidden_dim];
    for emb in term_embeddings.values() {
        for (i, &v) in emb.iter().enumerate() {
            mean_vec[i] += v;
        }
    }
    for v in mean_vec.iter_mut() {
        *v /= n_terms as f32;
    }

    let centred_embeddings: HashMap<String, Vec<f32>> = term_embeddings
        .iter()
        .map(|(term, emb)| {
            let centred: Vec<f32> = emb.iter()
                .zip(mean_vec.iter())
                .map(|(e, m)| e - m)
                .collect();
            (term.clone(), centred)
        })
        .collect();

    // Use centred embeddings as the embedding source for coherence analysis.
    let centred_source = PrecomputedEmbeddings::new(centred_embeddings)
        .expect("failed to build centred embeddings source");

    // Build Φ = I (identity).
    //
    // Detection uses z-scored logits (raw dot product h·u_i), not cos_Φ.
    // Pairwise analysis uses cos_I(u_i, u_j) = standard cosine, which
    // gives meaningful structure: bravery↔courage ≈ 0.76, efficiency↔tradition ≈ 0.24.
    //
    // The full UᵀU and term-focused EᵀE geometries both collapse —
    // all cos_Φ values are >0.98 with no discrimination.
    // Standard cosine in GPT-2's hidden space IS the model's geometry.
    eprintln!("Building Φ = I ({hidden_dim}×{hidden_dim}) for standard cosine pairwise...");
    let mut identity = vec![0.0f32; hidden_dim * hidden_dim];
    for i in 0..hidden_dim {
        identity[i * hidden_dim + i] = 1.0;
    }
    let geometry = CausalGeometry::from_raw_gram(identity, hidden_dim)
        .expect("identity matrix should be PD");
    eprintln!(
        "  positive definite: {}",
        geometry.is_positive_definite(),
    );

    // Load demo conversation
    let demo_conv_path = args.demo_conversation.as_deref()
        .unwrap_or("data/models/gpt2-demo-conversation.json");
    let demo_conversation_json = std::fs::read_to_string(demo_conv_path)
        .unwrap_or_else(|e| panic!("failed to read demo conversation {demo_conv_path}: {e}"));
    eprintln!("  demo conversation loaded from {demo_conv_path}");

    let mode = if args.values.is_some() { "taxonomy" } else { "gpt2" };

    AppState {
        geometry,
        term_embeddings,
        embedding_source: Box::new(centred_source),
        available_terms,
        hidden_dim,
        mode: mode.into(),
        demo_conversation_json,
        default_config: CoherenceConfig {
            antonym_threshold: -0.15,
            synonym_threshold: 0.20,
            severity_scale: Some(0.10),
        },
        introduction_threshold: 1.0,
        proxy: ProxyState::new(),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let state = build_state(&args);

    eprintln!("Mode: {} | hidden_dim: {} | terms: {}",
        state.mode, state.hidden_dim, state.available_terms.len());

    let state = Arc::new(state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Resolve static directory: CLI arg > crate-relative > cwd
    let static_dir = args.static_dir.unwrap_or_else(|| {
        // Try crate-relative path first (works during development)
        let crate_relative = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static");
        if crate_relative.exists() {
            crate_relative.to_string_lossy().to_string()
        } else {
            "static".to_string()
        }
    });
    eprintln!("Serving static files from: {static_dir}");

    let app = Router::new()
        .route("/api/demo-conversation", get(demo_conversation))
        .route("/api/conversation/analyse", post(api::analyse_conversation))
        .route("/api/embed", post(got_web::embed_api::embed_text))
        .route("/api/chat", post(got_web::chat_api::chat))
        // Proxy endpoints
        .route("/api/proxy/session", post(got_web::proxy_api::create_session))
        .route("/api/proxy/session/:id/observe", post(got_web::proxy_api::observe))
        .route("/api/proxy/session/:id/status", get(got_web::proxy_api::session_status))
        .route("/api/proxy/session/:id/history", get(got_web::proxy_api::deviation_history))
        .route("/api/proxy/session/:id/manifold", post(got_web::proxy_api::manifold))
        .route("/api/proxy/session/:id/snapshot", post(got_web::proxy_api::snapshot))
        .fallback_service(ServeDir::new(&static_dir))
        .layer(cors)
        .with_state(state);

    let addr: std::net::SocketAddr = args.listen.parse()
        .expect("invalid --listen address");
    eprintln!("got-web listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    axum::serve(listener, app)
        .await
        .expect("server error");
}
