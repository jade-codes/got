// ---------------------------------------------------------------------------
// got-web: conversational incoherence visualiser.
//
// A self-contained axum server that:
//   1. Serves a single-page D3.js frontend at GET /
//   2. Analyses conversation coherence at POST /api/conversation/analyse
//   3. Returns a demo conversation at GET /api/demo-conversation
//
// Two modes:
//   - Real model (--geometry path.gotue --vocab vocab.json):
//       loads GPT-2 (or any transformer) unembedding matrix,
//       builds Φ = I and uses vocabulary rows as value-term embeddings.
//   - Synthetic demo (--synthetic): compiled-in 32-d hand-crafted embeddings.
//       For development and testing only. Not credible for analysis.
// ---------------------------------------------------------------------------

mod api;
mod demo;

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
use tower_http::cors::{Any, CorsLayer};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "got-web", about = "Conversational incoherence visualiser")]
struct Args {
    /// Path to .gotue unembedding matrix file (enables real model mode).
    #[arg(long)]
    geometry: Option<String>,

    /// Path to vocabulary JSON array (required with --geometry).
    #[arg(long)]
    vocab: Option<String>,

    /// Path to demo conversation JSON with matching embedding dimensions.
    /// In real mode, defaults to data/models/gpt2-demo-conversation.json.
    #[arg(long)]
    demo_conversation: Option<String>,

    /// Listen address (default: 127.0.0.1:3000).
    #[arg(long, default_value = "127.0.0.1:3000")]
    listen: String,

    /// Run in synthetic demo mode (hand-crafted 32-d embeddings).
    /// For development/testing only — not credible for real analysis.
    #[arg(long)]
    synthetic: bool,
}

// Default value terms to look up in any model's vocabulary.
const VALUE_TERMS: &[&str] = &[
    "accountability", "bravery", "compassion", "courage", "cowardice",
    "creativity", "cruelty", "efficiency", "empathy", "equality",
    "equity", "fairness", "freedom", "honesty", "humility",
    "innovation", "integrity", "justice", "loyalty", "openness",
    "oppression", "resilience", "responsibility", "secrecy",
    "tradition", "transparency", "truthfulness", "wisdom",
];

/// Serve the single-page frontend.
async fn index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../static/index.html"),
    )
}

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
// State builders
// ---------------------------------------------------------------------------

/// Build state from compiled-in synthetic demo data.
fn build_synthetic_state() -> AppState {
    let term_embeddings: HashMap<String, Vec<f32>> =
        serde_json::from_str(demo::demo_embeddings_json())
            .expect("failed to parse demo embeddings");

    let dim = term_embeddings.values().next().unwrap().len();

    let geometry =
        api::build_geometry_from_embeddings(&term_embeddings, dim)
            .expect("failed to build demo geometry");

    let source = PrecomputedEmbeddings::from_json(demo::demo_embeddings_json())
        .expect("failed to load demo embeddings");

    let mut available_terms: Vec<String> = term_embeddings.keys().cloned().collect();
    available_terms.sort();

    AppState {
        geometry,
        term_embeddings,
        embedding_source: Box::new(source),
        available_terms,
        hidden_dim: dim,
        mode: "synthetic-demo".into(),
        demo_conversation_json: demo::demo_conversation_json().to_string(),
        default_config: CoherenceConfig {
            antonym_threshold: -0.5,
            synonym_threshold: 0.8,
            severity_scale: None,
        },
        introduction_threshold: 0.0,
    }
}

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
    // Bulk-read f32 LE values (much faster than individual byte conversions)
    let float_slice = &data[offset..offset + total_bytes];
    for chunk in float_slice.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }

    UnembeddingMatrix::new(vocab_size, hidden_dim, values)
        .expect("unembedding data length mismatch")
}

/// Build state from a real model's .gotue and vocabulary.
fn build_real_state(
    gotue_path: &str,
    vocab_path: &str,
    demo_conv_path: Option<&str>,
) -> AppState {
    eprintln!("Loading unembedding matrix from {gotue_path}...");
    let matrix = load_gotue(gotue_path);
    eprintln!(
        "  {} vocab × {} hidden dim",
        matrix.vocab_size, matrix.hidden_dim
    );

    let hidden_dim = matrix.hidden_dim;

    eprintln!("Loading vocabulary from {vocab_path}...");
    let vocab_json = std::fs::read_to_string(vocab_path)
        .unwrap_or_else(|e| panic!("failed to read {vocab_path}: {e}"));
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

    // Resolve the value terms to get their raw embeddings (for detection)
    let mut term_embeddings = HashMap::new();
    let mut available_terms = Vec::new();
    for &term in VALUE_TERMS {
        if let Some(emb) = lookup.embed(term) {
            term_embeddings.insert(term.to_string(), emb);
            available_terms.push(term.to_string());
        } else {
            eprintln!("  warning: term '{term}' not in vocabulary (skipped)");
        }
    }
    available_terms.sort();
    eprintln!("  resolved {}/{} value terms", available_terms.len(), VALUE_TERMS.len());

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

    // Load demo conversation for real mode
    let demo_conv_path = demo_conv_path
        .map(String::from)
        .unwrap_or_else(|| "data/models/gpt2-demo-conversation.json".to_string());
    let demo_conversation_json = std::fs::read_to_string(&demo_conv_path)
        .unwrap_or_else(|e| panic!("failed to read demo conversation {demo_conv_path}: {e}"));
    eprintln!("  demo conversation loaded from {demo_conv_path}");

    AppState {
        geometry,
        term_embeddings,
        embedding_source: Box::new(centred_source),
        available_terms,
        hidden_dim,
        mode: "gpt2".into(),
        demo_conversation_json,
        default_config: CoherenceConfig {
            antonym_threshold: -0.15,
            synonym_threshold: 0.20,
            severity_scale: Some(0.10),
        },
        introduction_threshold: 1.0,
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let state = if args.synthetic {
        eprintln!("*** SYNTHETIC DEMO MODE — NOT REAL MODEL DATA ***");
        eprintln!("This mode uses hand-crafted 32-d embeddings for development only.");
        build_synthetic_state()
    } else if let Some(ref gotue_path) = args.geometry {
        let vocab_path = args.vocab.as_deref()
            .expect("--vocab is required when --geometry is specified");
        build_real_state(
            gotue_path,
            vocab_path,
            args.demo_conversation.as_deref(),
        )
    } else {
        eprintln!("error: specify --geometry <path.gotue> --vocab <vocab.json> for real model mode,");
        eprintln!("       or --synthetic for hand-crafted demo data (development only).");
        std::process::exit(1);
    };

    eprintln!("Mode: {} | hidden_dim: {} | terms: {}",
        state.mode, state.hidden_dim, state.available_terms.len());

    let state = Arc::new(state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/demo-conversation", get(demo_conversation))
        .route("/api/conversation/analyse", post(api::analyse_conversation))
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
