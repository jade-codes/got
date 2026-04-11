// ---------------------------------------------------------------------------
// Hardware Capture Abstraction — Phase 11, §11.2–11.3.
//
// Defines the interface between GPU/accelerator hardware and the
// measurement enclave.  In a real deployment, this would be backed by
// DMA snooping, PCIe tap, or confidential-compute activation copy-out.
// For the PoC, `MockDmaTap` provides a test double.
// ---------------------------------------------------------------------------

use sha2::{Digest, Sha256};

/// A single activation frame captured from hardware.
///
/// Represents the raw layer output buffer copied from VRAM (or equivalent)
/// by the hardware tap.  The `integrity_hash` is computed by the capture
/// hardware itself — the model process never touches it.
#[derive(Debug, Clone)]
pub struct ActivationFrame {
    /// Which layer produced this activation.
    pub layer: usize,
    /// Token position (or batch index) this frame corresponds to.
    pub token_position: usize,
    /// Raw activation values — f32 residual-stream vector (length = hidden_dim).
    pub values: Vec<f32>,
    /// SHA-256 of the raw f32 LE bytes, computed by the capture hardware.
    /// The enclave recomputes this to verify the frame was not tampered
    /// with in transit from capture hardware to enclave.
    pub integrity_hash: [u8; 32],
}

impl ActivationFrame {
    /// Compute the expected integrity hash from the activation values,
    /// layer index, and token position.
    ///
    /// Including layer and position prevents frame-swap attacks where an
    /// adversary reorders captured frames between layers.
    pub fn compute_hash(layer: usize, token_position: usize, values: &[f32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(layer.to_le_bytes());
        hasher.update(token_position.to_le_bytes());
        for v in values {
            // Canonicalise: -0.0 → 0.0 to match the convention used by
            // serialise_for_signing, geometry_hash, and compute_input_hash.
            let canon = if *v == 0.0 { 0.0f32 } else { *v };
            hasher.update(canon.to_le_bytes());
        }
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Verify that the integrity hash matches the activation data.
    pub fn verify_integrity(&self) -> bool {
        let expected = Self::compute_hash(self.layer, self.token_position, &self.values);
        self.integrity_hash == expected
    }
}

/// Abstraction over hardware activation capture.
///
/// **A real implementation IS the byte path** between the model's
/// memory and the enclave's memory — every byte the enclave consumes
/// flows through this trait.  The integrity hash on each
/// `ActivationFrame` MUST be computed by the capture hardware itself,
/// not by the model process; the enclave then recomputes the hash via
/// `frame.verify_integrity()` to detect tampering in transit.
///
/// Per-platform sketches:
///   - **GPU DMA snoop**: capture layer output buffers from VRAM
///     before the model process can sanitise them.
///   - **NVIDIA H100 Confidential Computing**: the natural fit —
///     capture and enclave both live inside the GPU's CC partition,
///     activations are read directly from encrypted VRAM, no CPU
///     round-trip.
///   - **AMD SEV-SNP**: model process and capture process both run
///     inside the encrypted guest VM; the hypervisor cannot read
///     guest memory.
///   - **Intel SGX**: harder, since the model usually runs on a GPU
///     outside the enclave; SGX2 with EDMM can run a CPU model
///     entirely inside the enclave at the cost of throughput.
///
/// The mock (`MockDmaTap`) computes hashes in-process and is
/// intentionally insecure — it exists only so tests can exercise the
/// pipeline without real hardware.  See
/// `docs/enclave-adapter-contract.md` for the full contract.
///
/// The trait is `Send + Sync` so implementations can run in dedicated
/// capture threads.
pub trait HardwareCapture: Send + Sync {
    /// Capture activations for a given layer and token position.
    ///
    /// Returns an `ActivationFrame` with an integrity hash computed by
    /// the capture hardware.  Returns `None` if the layer/position is
    /// not configured for capture.
    fn capture(
        &self,
        layer: usize,
        token_position: usize,
        values: &[f32],
    ) -> Option<ActivationFrame>;

    /// The hidden dimension this capture device expects.
    fn hidden_dim(&self) -> usize;

    /// Which layers are being captured.
    fn captured_layers(&self) -> Vec<usize>;
}

// ---------------------------------------------------------------------------
// MockDmaTap — test double for hardware activation capture
// ---------------------------------------------------------------------------

/// A mock DMA tap that simulates hardware capture.
///
/// Computes honest integrity hashes and optionally injects bit-flips
/// to test tamper detection.
#[derive(Debug, Clone)]
pub struct MockDmaTap {
    /// Hidden dimension of the model.
    hidden_dim: usize,
    /// Which layers to capture (empty = all).
    layers: Vec<usize>,
    /// If true, corrupt the integrity hash to simulate tampered data.
    inject_tamper: bool,
}

impl MockDmaTap {
    /// Create a new mock DMA tap.
    pub fn new(hidden_dim: usize, layers: Vec<usize>) -> Self {
        Self {
            hidden_dim,
            layers,
            inject_tamper: false,
        }
    }

    /// Enable integrity-hash corruption for tamper testing.
    pub fn with_tamper(mut self) -> Self {
        self.inject_tamper = true;
        self
    }
}

impl HardwareCapture for MockDmaTap {
    fn capture(
        &self,
        layer: usize,
        token_position: usize,
        values: &[f32],
    ) -> Option<ActivationFrame> {
        // Only capture configured layers (empty = capture all).
        if !self.layers.is_empty() && !self.layers.contains(&layer) {
            return None;
        }

        let mut integrity_hash = ActivationFrame::compute_hash(layer, token_position, values);

        if self.inject_tamper {
            // Flip a bit in the hash to simulate hardware-level tampering.
            integrity_hash[0] ^= 0xFF;
        }

        Some(ActivationFrame {
            layer,
            token_position,
            values: values.to_vec(),
            integrity_hash,
        })
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn captured_layers(&self) -> Vec<usize> {
        self.layers.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_frame_integrity_valid() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let hash = ActivationFrame::compute_hash(0, 0, &values);
        let frame = ActivationFrame {
            layer: 0,
            token_position: 0,
            values: values.clone(),
            integrity_hash: hash,
        };
        assert!(frame.verify_integrity());
    }

    #[test]
    fn activation_frame_integrity_tampered() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut hash = ActivationFrame::compute_hash(0, 0, &values);
        hash[0] ^= 0x01; // flip a bit
        let frame = ActivationFrame {
            layer: 0,
            token_position: 0,
            values,
            integrity_hash: hash,
        };
        assert!(!frame.verify_integrity());
    }

    #[test]
    fn activation_frame_hash_deterministic() {
        #[allow(clippy::approx_constant)]
        let values = vec![1.0f32, -0.5, 3.14159, 0.0];
        let h1 = ActivationFrame::compute_hash(0, 0, &values);
        let h2 = ActivationFrame::compute_hash(0, 0, &values);
        assert_eq!(h1, h2);
    }

    #[test]
    fn mock_dma_captures_configured_layers() {
        let tap = MockDmaTap::new(4, vec![0, 2]);
        let h = vec![1.0, 2.0, 3.0, 4.0];

        assert!(tap.capture(0, 0, &h).is_some());
        assert!(tap.capture(1, 0, &h).is_none()); // layer 1 not configured
        assert!(tap.capture(2, 0, &h).is_some());
    }

    #[test]
    fn mock_dma_captures_all_when_empty() {
        let tap = MockDmaTap::new(4, vec![]);
        let h = vec![1.0, 2.0, 3.0, 4.0];

        assert!(tap.capture(0, 0, &h).is_some());
        assert!(tap.capture(5, 0, &h).is_some());
        assert!(tap.capture(99, 0, &h).is_some());
    }

    #[test]
    fn mock_dma_honest_hash() {
        let tap = MockDmaTap::new(4, vec![]);
        let h = vec![1.0, 2.0, 3.0, 4.0];
        let frame = tap.capture(0, 0, &h).unwrap();
        assert!(frame.verify_integrity());
    }

    #[test]
    fn mock_dma_tampered_hash() {
        let tap = MockDmaTap::new(4, vec![]).with_tamper();
        let h = vec![1.0, 2.0, 3.0, 4.0];
        let frame = tap.capture(0, 0, &h).unwrap();
        assert!(
            !frame.verify_integrity(),
            "tampered hash should fail integrity check"
        );
    }

    #[test]
    fn mock_dma_hidden_dim() {
        let tap = MockDmaTap::new(768, vec![0, 1]);
        assert_eq!(tap.hidden_dim(), 768);
        assert_eq!(tap.captured_layers(), vec![0, 1]);
    }
}
