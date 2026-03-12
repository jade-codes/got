// ---------------------------------------------------------------------------
// Hardware-Isolated Measurement — Phase 11.
//
// Moves activation capture to hardware that the model cannot influence.
// The measurement enclave receives raw activations via a hardware tap
// (DMA / secure copy), runs causal probes, and produces signed
// attestations.  The signing key never leaves the enclave.
//
// This crate defines:
//   - `HardwareCapture` trait — abstraction over DMA tap / activation copy
//   - `MeasurementEnclave` trait — isolated measurement environment
//   - `MockDmaTap` — test double for hardware capture
//   - `MockEnclave` — in-process enclave for unit and integration tests
// ---------------------------------------------------------------------------

pub mod capture;
pub mod enclave;

pub use capture::{ActivationFrame, HardwareCapture, MockDmaTap};
pub use enclave::{EnclaveError, MeasurementEnclave};

#[cfg(any(test, feature = "mock"))]
pub use enclave::{enclave_pipeline, MockEnclave, MockEnclaveConfig};
