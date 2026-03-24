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
use got_incoherence::coherence::CoherenceConfig;
use got_incoherence::embeddings::PrecomputedEmbeddings;
use got_web::AppState;
use got_web::VocabLookup;
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
    geometry: Option<String>,

    /// Path to vocabulary JSON array (required with --geometry).
    #[arg(long)]
    vocab: Option<String>,

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

    /// URL of the activation server for intermediate-layer hidden states.
    /// When set, /api/embed routes through the sidecar for real residual stream activations.
    /// Example: http://localhost:8100
    #[arg(long)]
    activation_server: Option<String>,

    /// Run in synthetic demo mode (compiled-in 32-d embeddings).
    /// For deployment/development without a reference model.
    #[arg(long)]
    synthetic: bool,

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
// State builders
// ---------------------------------------------------------------------------

/// Build state from compiled-in synthetic demo data (for deployment without a reference model).
fn build_synthetic_state() -> AppState {
    let term_embeddings: HashMap<String, Vec<f32>> =
        serde_json::from_str(demo::demo_embeddings_json())
            .expect("failed to parse demo embeddings");

    let dim = term_embeddings.values().next().unwrap().len();

    let geometry = CausalGeometry::identity(dim);

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
        proxy: ProxyState::new(),
        vocab_lookup: None,
        activation_server_url: None,
    }
}

/// Build application state from the reference model and value terms.
fn build_state(args: &Args) -> AppState {
    let gotue_path = args.geometry.as_deref()
        .expect("--geometry is required when not using --synthetic");
    let vocab_path = args.vocab.as_deref()
        .expect("--vocab is required when not using --synthetic");

    eprintln!("Loading value terms from {gotue_path} (selective)...");

    // Collect all tokens we need to look up
    let mut needed_tokens: Vec<String> = Vec::new();

    let taxonomy = if let Some(ref values_path) = args.values {
        eprintln!("Loading value taxonomy from {values_path}...");
        let toml_str = std::fs::read_to_string(values_path)
            .unwrap_or_else(|e| panic!("failed to read {values_path}: {e}"));
        let tax: ValueTaxonomy = toml::from_str(&toml_str)
            .unwrap_or_else(|e| panic!("failed to parse taxonomy TOML: {e}"));
        eprintln!("  {} value entries loaded", tax.values.len());

        // Collect all tokens from descriptions
        for entry in &tax.values {
            for token in entry.description.split_whitespace() {
                let clean: String = token.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                    .collect();
                if !clean.is_empty() {
                    needed_tokens.push(clean);
                }
            }
        }
        Some(tax)
    } else {
        // Fallback: single-token lookup
        for &term in DEFAULT_VALUE_TERMS {
            needed_tokens.push(term.to_string());
        }
        None
    };

    needed_tokens.sort();
    needed_tokens.dedup();

    // Load .gotue file and build full vocab lookup.
    // The file stays in memory for on-demand row reads (embed endpoint).
    // Load .gotue file — stays in memory for the VocabLookup
    let gotue_data = std::fs::read(gotue_path)
        .unwrap_or_else(|e| panic!("failed to read {gotue_path}: {e}"));
    if gotue_data.len() < 14 || &gotue_data[0..4] != b"GOTU" {
        panic!("{gotue_path}: not a valid .gotue file");
    }
    let vocab_size = u32::from_le_bytes(gotue_data[6..10].try_into().unwrap()) as usize;
    let hidden_dim = u32::from_le_bytes(gotue_data[10..14].try_into().unwrap()) as usize;
    let gotue_data_start = 14;
    eprintln!("  {} vocab × {} hidden dim", vocab_size, hidden_dim);

    // Build full vocab index
    let vocab_json = std::fs::read_to_string(vocab_path)
        .unwrap_or_else(|e| panic!("failed to read {vocab_path}: {e}"));
    let vocab_raw: Vec<String> = serde_json::from_str(&vocab_json)
        .unwrap_or_else(|e| panic!("failed to parse vocab JSON: {e}"));

    let mut vocab_index: HashMap<String, usize> = HashMap::with_capacity(vocab_size);
    for (idx, tok) in vocab_raw.iter().enumerate() {
        if idx >= vocab_size { break; }
        let clean = tok.replace('Ġ', "").to_lowercase();
        vocab_index.entry(clean).or_insert(idx);
    }
    eprintln!("  vocabulary index: {} entries", vocab_index.len());

    // Build VocabLookup for the embed endpoint
    let vocab_lookup = VocabLookup {
        index: vocab_index,
        data: gotue_data,
        data_start: gotue_data_start,
        hidden_dim,
    };

    // Extract needed token embeddings for value terms
    let mut token_embeddings = HashMap::new();
    for token in &needed_tokens {
        if let Some(emb) = vocab_lookup.embed(token) {
            token_embeddings.insert(token.clone(), emb);
        }
    }
    eprintln!("  loaded {}/{} token embeddings", token_embeddings.len(), needed_tokens.len());

    // Resolve value terms to embeddings
    let mut term_embeddings = HashMap::new();
    let mut available_terms = Vec::new();

    if let Some(ref tax) = taxonomy {
        for entry in &tax.values {
            // Average token embeddings from the description
            let mut sum = vec![0.0f32; hidden_dim];
            let mut matched = 0usize;
            let total = entry.description.split_whitespace().count();
            for token in entry.description.split_whitespace() {
                let clean: String = token.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                    .collect();
                if let Some(emb) = token_embeddings.get(&clean) {
                    for (s, e) in sum.iter_mut().zip(emb.iter()) { *s += e; }
                    matched += 1;
                }
            }
            if matched > 0 {
                let scale = 1.0 / matched as f32;
                for s in sum.iter_mut() { *s *= scale; }
                eprintln!("  '{}': {}/{} tokens matched", entry.name, matched, total);
                term_embeddings.insert(entry.name.clone(), sum);
                available_terms.push(entry.name.clone());
            } else {
                eprintln!("  warning: '{}' — no tokens matched (skipped)", entry.name);
            }
        }
    } else {
        for &term in DEFAULT_VALUE_TERMS {
            if let Some(emb) = token_embeddings.get(term) {
                term_embeddings.insert(term.to_string(), emb.clone());
                available_terms.push(term.to_string());
            } else {
                eprintln!("  warning: term '{term}' not found (skipped)");
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

    // Compute Φ = UᵀU — the causal inner product.
    // Load the full unembedding matrix, compute the d×d Gram matrix, then drop U.
    eprintln!("Computing Φ = UᵀU ({hidden_dim}×{hidden_dim}) from {gotue_path}...");
    let geometry = {
        let data = std::fs::read(gotue_path)
            .unwrap_or_else(|e| panic!("failed to read {gotue_path}: {e}"));
        let mut offset = 6; // skip magic + version
        let vocab_size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let hd = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        assert_eq!(hd, hidden_dim);

        let total_bytes = vocab_size * hidden_dim * 4;
        let float_data = &data[offset..offset + total_bytes];

        // Build faer matrix U (V × d) and compute Φ = UᵀU (d × d)
        let u_mat = faer::Mat::from_fn(vocab_size, hidden_dim, |i, j| {
            let idx = (i * hidden_dim + j) * 4;
            f32::from_le_bytes(float_data[idx..idx + 4].try_into().unwrap()) as f64
        });
        eprintln!("  U loaded: {} × {}, computing UᵀU...", vocab_size, hidden_dim);
        let phi = u_mat.transpose() * &u_mat; // d × d
        eprintln!("  UᵀU computed");

        // Convert to flat f32 row-major
        let mut gram = vec![0.0f32; hidden_dim * hidden_dim];
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                gram[i * hidden_dim + j] = phi.read(i, j) as f32;
            }
        }

        // U is dropped here — only the d×d Gram matrix is kept
        CausalGeometry::from_raw_gram(gram, hidden_dim)
            .expect("UᵀU should be positive semi-definite")
    };
    eprintln!("  positive definite: {}", geometry.is_positive_definite());

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
        vocab_lookup: Some(vocab_lookup),
        activation_server_url: args.activation_server.clone(),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let state = if args.synthetic {
        build_synthetic_state()
    } else if args.geometry.is_some() {
        build_state(&args)
    } else {
        eprintln!("error: specify --geometry <path.gotue> --vocab <vocab.json>, or --synthetic");
        std::process::exit(1);
    };

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
        // Metrics endpoints
        .route("/api/coherence", post(got_web::metrics_api::coherence))
        .route("/api/collapse", post(got_web::metrics_api::collapse))
        .route("/api/compare", post(got_web::metrics_api::compare))
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
