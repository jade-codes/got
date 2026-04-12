// ---------------------------------------------------------------------------
// HttpSyncSource — pull-based federation refresh over HTTP(S).
//
// Implements `got_wire::federation::FederationSyncSource` using
// `reqwest::blocking`, sending `If-None-Match: "<previous_digest_hex>"`
// when the caller already holds a digest.  A `304 Not Modified`
// response is mapped to `Ok(None)` (cheap "nothing changed" path).
// A `200 OK` response computes the body's SHA-256 — that becomes the
// fetched digest, no matter what the server's `ETag` header says.
//
// Why blocking instead of async reqwest:
//
// `FederationSyncSource::fetch` is sync because the trait is sync, and
// the trait is sync because every other implementation (file, static,
// in-memory) is sync.  `FederationSyncManager` runs every fetch
// inside `tokio::task::spawn_blocking`, so a blocking HTTP client is
// already isolated from the async runtime.  `reqwest::blocking`
// internally manages its own short-lived runtime per call, which is
// fine here because spawn_blocking already gives us a dedicated
// thread.
//
// Why we ignore the server's ETag value for digest computation:
//
// We could trust the server's `ETag` header to identify the resource,
// but the digest the rest of the protocol uses must be a true SHA-256
// of the bytes — not "whatever string the server returned in ETag".
// We do *send* the etag back via `If-None-Match` to take advantage of
// the cheap 304 path, but we *recompute* the digest from the body
// bytes on every 200 response.
// ---------------------------------------------------------------------------

use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, IF_NONE_MATCH};
use reqwest::StatusCode;

use got_wire::federation::{FederationSyncSource, SyncedRegistry};
use got_wire::WireError;

/// Pull-based HTTP `FederationSyncSource`.
///
/// Construct with `new(name, url)` and pass to a
/// `FederationSyncManager` like any other source.  The default
/// timeout is 30 seconds; override with `with_timeout`.
pub struct HttpSyncSource {
    name: String,
    url: String,
    client: Client,
}

impl std::fmt::Debug for HttpSyncSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpSyncSource")
            .field("name", &self.name)
            .field("url", &self.url)
            .finish()
    }
}

impl HttpSyncSource {
    /// Construct an `HttpSyncSource` with default settings (30s
    /// timeout, rustls TLS).  Returns an error only if the
    /// underlying reqwest client fails to build, which in practice
    /// only happens on platform setup issues.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Result<Self, WireError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("got-net/0.1")
            .build()
            .map_err(|e| WireError::Io(format!("reqwest build: {e}")))?;
        Ok(Self {
            name: name.into(),
            url: url.into(),
            client,
        })
    }

    /// Override the request timeout.  Pass after `new()`.
    pub fn with_timeout(self, timeout: Duration) -> Result<Self, WireError> {
        let client = Client::builder()
            .timeout(timeout)
            .user_agent("got-net/0.1")
            .build()
            .map_err(|e| WireError::Io(format!("reqwest build: {e}")))?;
        Ok(Self { client, ..self })
    }
}

impl FederationSyncSource for HttpSyncSource {
    fn fetch(&self, since: Option<[u8; 32]>) -> Result<Option<SyncedRegistry>, WireError> {
        let mut headers = HeaderMap::new();
        if let Some(digest) = since {
            // RFC 7232 says ETag values are quoted opaque strings.
            // We use the hex digest as the etag value.
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            let value = format!("\"{hex}\"");
            if let Ok(hv) = HeaderValue::from_str(&value) {
                headers.insert(IF_NONE_MATCH, hv);
            }
        }

        let response = self
            .client
            .get(&self.url)
            .headers(headers)
            .send()
            .map_err(|e| WireError::Io(format!("http fetch {}: {e}", self.url)))?;

        let status = response.status();
        if status == StatusCode::NOT_MODIFIED {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(WireError::Io(format!(
                "http fetch {}: {} {}",
                self.url,
                status.as_u16(),
                status.canonical_reason().unwrap_or("")
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| WireError::Io(format!("http body {}: {e}", self.url)))?
            .to_vec();
        if bytes.len() > super::transport::MAX_MESSAGE_SIZE {
            return Err(WireError::Io(format!(
                "http body {}: {} bytes exceeds limit {}",
                self.url,
                bytes.len(),
                super::transport::MAX_MESSAGE_SIZE
            )));
        }
        let fetched_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(Some(SyncedRegistry::from_bytes(bytes, fetched_at)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_default_source() {
        let src = HttpSyncSource::new("eu", "https://example.invalid/registry.toml").unwrap();
        assert_eq!(src.name(), "eu");
    }

    #[test]
    fn fetch_unreachable_url_returns_io_error() {
        // Use TEST-NET-1 (RFC 5737) which is documented as
        // unallocated for documentation use; reqwest should fail
        // fast with a connect error rather than serving a real
        // response.
        let src = HttpSyncSource::new("test", "http://192.0.2.1:1/registry.toml")
            .unwrap()
            .with_timeout(Duration::from_millis(500))
            .unwrap();
        let err = src.fetch(None).unwrap_err();
        assert!(matches!(err, WireError::Io(_)));
    }

    /// End-to-end test against a real localhost HTTP server.  Uses
    /// the standard library's `TcpListener` to accept one request,
    /// inspects the headers, returns a known body, and confirms the
    /// `HttpSyncSource` correctly handles both the initial fetch and
    /// the `If-None-Match` 304 path.
    #[test]
    fn end_to_end_against_local_http_server() {
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        // Server thread: handles up to 2 requests, then exits.
        // Request 1: no If-None-Match → 200 with body.
        // Request 2: If-None-Match present → 304.
        let server = thread::spawn(move || {
            for expect_inm in [false, true] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut saw_inm = false;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if line.to_ascii_lowercase().starts_with("if-none-match:") {
                        saw_inm = true;
                    }
                }
                let _ = expect_inm; // we just record what we saw
                if saw_inm {
                    let response = "HTTP/1.1 304 Not Modified\r\n\r\n";
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    let body = b"version = 1\n";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                    stream.write_all(body).unwrap();
                }
                let _ = stream.flush();
                drop(reader);
                drop(stream);
                let _ = expect_inm;
            }
        });

        let url = format!("http://127.0.0.1:{port}/registry.toml");
        let src = HttpSyncSource::new("eu", &url)
            .unwrap()
            .with_timeout(Duration::from_secs(2))
            .unwrap();

        // First fetch: returns the body.
        let first = src
            .fetch(None)
            .unwrap()
            .expect("first fetch returns content");
        assert_eq!(first.bytes, b"version = 1\n");

        // Second fetch with the same digest: server returns 304.
        let second = src.fetch(Some(first.digest)).unwrap();
        assert!(second.is_none());

        server.join().unwrap();
    }
}
