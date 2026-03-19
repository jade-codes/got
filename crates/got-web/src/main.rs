// ---------------------------------------------------------------------------
// got-web: conversational incoherence visualiser.
//
// A self-contained axum server that:
//   1. Serves a single-page D3.js frontend at GET /
//   2. Analyses conversation coherence at POST /api/conversation/analyse
//   3. Returns a demo conversation at GET /api/demo-conversation
//
// All data (demo embeddings + HTML frontend) is compiled into the binary
// via include_str! — zero external files at runtime.
// ---------------------------------------------------------------------------

mod api;
mod demo;

use axum::{
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

/// Serve the single-page frontend.
async fn index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../static/index.html"),
    )
}

/// Return the demo conversation (pre-built scenario with message embeddings).
async fn demo_conversation() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        demo::demo_conversation_json(),
    )
}

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/demo-conversation", get(demo_conversation))
        .route("/api/conversation/analyse", post(api::analyse_conversation))
        .layer(cors);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("got-web listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    axum::serve(listener, app)
        .await
        .expect("server error");
}
