// ---------------------------------------------------------------------------
// TcpTransport — sync `got_wire::noise::Transport` impl over a real socket.
//
// The wire `Transport` trait is byte-stream-oriented: `send` writes raw
// bytes, `recv(len)` reads exactly `len` bytes.  Length-prefixing for
// individual messages happens inside `NoiseSession::send_encrypted` /
// `recv_encrypted`, not here — this transport just blits bytes onto the
// socket and pulls them off again.  That keeps `TcpTransport` a thin
// adapter and matches the contract `MemoryTransport` already implements
// for the in-memory test path.
//
// `MAX_READ_SIZE` (16 MiB) caps a single `recv` call to bound the
// maximum allocation a malicious peer can dictate per call.
// ---------------------------------------------------------------------------

use std::io::{Read, Write};
use std::net::TcpStream;

use got_wire::noise::Transport;
use got_wire::WireError;

/// Maximum bytes returned from a single `recv(len)` call.
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Sync TCP transport implementing `got_wire::noise::Transport`.  Slots
/// directly into the existing Noise NK handshake and the encrypted
/// `NoiseSession::send_encrypted` / `recv_encrypted` paths — those
/// internally length-prefix their messages, so this transport only
/// needs to be a faithful byte-stream pipe.
pub struct TcpTransport {
    stream: TcpStream,
}

impl TcpTransport {
    /// Wrap an already-connected `TcpStream`.  The caller is responsible
    /// for any socket-level options (timeouts, nodelay) before handing
    /// the stream over.
    pub fn new(stream: TcpStream) -> Self {
        Self { stream }
    }

    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }

    pub fn stream_mut(&mut self) -> &mut TcpStream {
        &mut self.stream
    }

    pub fn into_inner(self) -> TcpStream {
        self.stream
    }
}

impl Transport for TcpTransport {
    fn send(&mut self, data: &[u8]) -> Result<(), WireError> {
        self.stream
            .write_all(data)
            .map_err(|e| WireError::Io(e.to_string()))?;
        self.stream
            .flush()
            .map_err(|e| WireError::Io(e.to_string()))?;
        Ok(())
    }

    fn recv(&mut self, len: usize) -> Result<Vec<u8>, WireError> {
        if len > MAX_MESSAGE_SIZE {
            return Err(WireError::PayloadTooLarge {
                size: len as u32,
                limit: MAX_MESSAGE_SIZE as u32,
            });
        }
        let mut data = vec![0u8; len];
        self.stream
            .read_exact(&mut data)
            .map_err(|e| WireError::Io(e.to_string()))?;
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    #[test]
    fn raw_byte_stream_roundtrip() {
        // Validate that send/recv round-trip arbitrary byte sequences.
        // The Noise layer above will do its own length-prefixing.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server = thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut t = TcpTransport::new(sock);
            // Read a 4-byte header then a 5-byte body — exactly how
            // NoiseSession::recv_encrypted does it.
            let header = t.recv(4).unwrap();
            assert_eq!(header, [0u8, 0, 0, 5]);
            let body = t.recv(5).unwrap();
            assert_eq!(body, b"hello");
            t.send(&[0u8, 0, 0, 5]).unwrap();
            t.send(b"world").unwrap();
        });

        let client = TcpStream::connect(addr).unwrap();
        let mut t = TcpTransport::new(client);
        t.send(&[0u8, 0, 0, 5]).unwrap();
        t.send(b"hello").unwrap();
        let header = t.recv(4).unwrap();
        assert_eq!(header, [0u8, 0, 0, 5]);
        let body = t.recv(5).unwrap();
        assert_eq!(body, b"world");

        server.join().unwrap();
    }

    #[test]
    fn rejects_oversized_recv() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _server = thread::spawn(move || {
            let _ = listener.accept().unwrap();
            // Hold the connection open while the client tests its
            // recv-size guard.  Drop drops the socket on join.
            thread::sleep(std::time::Duration::from_millis(100));
        });
        let client = TcpStream::connect(addr).unwrap();
        let mut t = TcpTransport::new(client);
        let err = t.recv(MAX_MESSAGE_SIZE + 1).unwrap_err();
        assert!(matches!(err, WireError::PayloadTooLarge { .. }));
    }
}
