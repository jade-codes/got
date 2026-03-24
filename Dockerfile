# Multi-stage build for got-web
# Stage 1: Build the Rust binary
FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY . .

RUN cargo build --release -p got-web

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/got-web /usr/local/bin/got-web
COPY --from=builder /app/crates/got-web/static /app/static
COPY --from=builder /app/data /app/data

WORKDIR /app

# Cloud Run sets PORT env var
ENV PORT=8080

EXPOSE 8080

# Run in synthetic mode — Cloud Run sets PORT env var
CMD ["sh", "-c", "got-web --synthetic --listen 0.0.0.0:${PORT} --static-dir /app/static"]
