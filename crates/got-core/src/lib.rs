pub mod geometry;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Serde helpers for fixed-size byte arrays (serde doesn't derive for [u8; N>32])
// ---------------------------------------------------------------------------

/// Serde helper for `[u8; 32]` — serialises as hex string.
pub mod hex32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 hex chars for 32 bytes, got {}",
                s.len()
            )));
        }
        // Validate all bytes are ASCII hex digits before indexing.
        // This prevents panics from str slicing at non-char-boundary
        // byte offsets when the string contains multi-byte UTF-8.
        if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom(
                "non-hex characters in 32-byte hash field",
            ));
        }
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(serde::de::Error::custom))
            .collect::<Result<Vec<_>, _>>()?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

/// Serde helper for `[u8; 64]` — serialises as hex string.
pub mod hex64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 128 {
            return Err(serde::de::Error::custom(format!(
                "expected 128 hex chars for 64 bytes, got {}",
                s.len()
            )));
        }
        // Validate all bytes are ASCII hex digits before indexing.
        // This prevents panics from str slicing at non-char-boundary
        // byte offsets when the string contains multi-byte UTF-8.
        if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom(
                "non-hex characters in 64-byte signature field",
            ));
        }
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(serde::de::Error::custom))
            .collect::<Result<Vec<_>, _>>()?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 bytes"))
    }
}

/// Serde helper for `Option<[u8; 32]>` — serialises as hex string when present.
pub mod optional_hex32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<[u8; 32]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(bytes) => {
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                serializer.serialize_some(&hex)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            None => Ok(None),
            Some(s) => {
                if s.len() != 64 {
                    return Err(serde::de::Error::custom(format!(
                        "expected 64 hex chars for 32 bytes, got {}",
                        s.len()
                    )));
                }
                if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(serde::de::Error::custom(
                        "non-hex characters in optional 32-byte hash field",
                    ));
                }
                let bytes: Vec<u8> = (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(serde::de::Error::custom))
                    .collect::<Result<Vec<_>, _>>()?;
                let arr: [u8; 32] = bytes
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("expected 32 bytes"))?;
                Ok(Some(arr))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Precision & inner‐product enums
// ---------------------------------------------------------------------------

/// Numerical precision the activations were extracted at.
/// Attestation comparison is only valid between matching precisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Bfloat16,
    Int8,
}

impl Precision {
    pub fn tag(self) -> u8 {
        match self {
            Self::Fp32 => 0,
            Self::Fp16 => 1,
            Self::Bfloat16 => 2,
            Self::Int8 => 3,
        }
    }
}

/// Which inner product was used for probe training and inference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InnerProduct {
    Causal,
    Euclidean,
    CausalRegularised { epsilon: f32 },
}

impl InnerProduct {
    pub fn tag(self) -> u8 {
        match self {
            Self::Causal => 0,
            Self::Euclidean => 1,
            Self::CausalRegularised { .. } => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Attestation schema (Section 6)
// ---------------------------------------------------------------------------

/// The attestation schema.
/// All fields required. Invalid if signature does not verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricAttestation {
    /// Wire format version — always first in serialised form.
    pub schema_version: u16,
    pub model_id: String,
    /// Merkle root over weight shards (sorted lexicographically by tensor name).
    /// `None` when model shards were not provided at attestation time.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hex32"
    )]
    pub model_hash: Option<[u8; 32]>,
    pub precision: Precision,
    pub inner_product: InnerProduct,
    /// SHA-256 of the activation input file (covers model ID, precision, and all activation data).
    #[serde(with = "hex32")]
    pub input_hash: [u8; 32],
    /// Unix UTC seconds. Included in signature payload.
    pub timestamp: u64,
    pub corpus_version: String,
    pub probe_version: String,
    /// Probe readings per layer: layer_readings[layer_idx][dim_idx].
    pub layer_readings: Vec<Vec<f32>>,
    /// Platt-scaled confidence per dimension (flattened across layers, in order).
    pub confidence: Vec<f32>,
    /// true = below reliability threshold for that dimension.
    pub coverage_flags: Vec<bool>,
    pub divergence_flag: bool,

    // --- v2 chained attestation fields (None for v1) ---
    /// SHA-256 of the serialised parent attestation. None for epoch 0 or v1.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hex32"
    )]
    pub parent_attestation_hash: Option<[u8; 32]>,
    /// SHA-256 of the Gram matrix Φ at time of this attestation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hex32"
    )]
    pub geometry_hash: Option<[u8; 32]>,
    /// Normalised Frobenius drift from the reference geometry. 0.0 if unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_drift: Option<f32>,

    // --- v3 causal intervention fields (empty/None for v1/v2) ---
    /// Per-probe causal intervention results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub causal_scores: Vec<CausalScoreRecord>,
    /// The δ perturbation magnitude used for causal intervention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_delta: Option<f32>,
    /// All probes passed causal check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_flag: Option<bool>,

    // --- v3 adversarial hardening fields (Phase 13) ---
    /// Monotonic sequence number assigned by the enclave.
    /// The enclave increments this on every attestation; the counter
    /// never resets. Gaps in the sequence indicate omitted attestations.
    #[serde(default)]
    pub sequence_number: u64,

    /// Per-probe directional drift records. Empty for pre-Phase 13 attestations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directional_drifts: Vec<DirectionalDrift>,

    /// SHA-256 commitment to the sampled probe indices, chosen before the
    /// model sees any activations for this window.  In a real TEE the
    /// commitment is published before activations are captured; verifiers
    /// can check that the probes actually run match the commitment.
    /// None for pre-Phase-13 attestations.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hex32"
    )]
    pub probe_commitment: Option<[u8; 32]>,

    /// Ed25519 over all preceding fields (canonical serialisation).
    #[serde(with = "hex64")]
    pub signature: [u8; 64],
}

/// Newtype wrapper that marks an attestation as **unsigned**.
///
/// Exists to prevent accidental use of an unsigned `GeometricAttestation`
/// in verification, exchange, or storage code.  `assemble_and_sign()`
/// consumes an `UnsignedAttestation` and returns the signed
/// `GeometricAttestation`.  Call `.into_inner()` only when you need the
/// raw struct for serialisation before signing (this is what
/// `got_attest::assemble_and_sign` does internally).
#[derive(Debug, Clone)]
pub struct UnsignedAttestation(pub GeometricAttestation);

impl UnsignedAttestation {
    /// Consume the wrapper and return the inner `GeometricAttestation`.
    pub fn into_inner(self) -> GeometricAttestation {
        self.0
    }
}

impl From<GeometricAttestation> for UnsignedAttestation {
    fn from(a: GeometricAttestation) -> Self {
        Self(a)
    }
}

// ---------------------------------------------------------------------------
// Activation data types
// ---------------------------------------------------------------------------

/// Residual stream activations at one layer for one token position.
/// Extracted externally (Python) and loaded here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivation {
    pub layer: usize,
    pub token_position: usize,
    /// length = hidden_dim d
    pub values: Vec<f32>,
}

/// The unembedding matrix U ∈ ℝ^{V × d}, row-major.
/// Used to compute the Gram matrix Φ = UᵀU for the causal inner product.
#[derive(Debug, Clone)]
pub struct UnembeddingMatrix {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    /// Row-major data, length = vocab_size × hidden_dim.
    pub data: Vec<f32>,
}

impl UnembeddingMatrix {
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        data: Vec<f32>,
    ) -> Result<Self, geometry::GeometryError> {
        if data.len() != vocab_size * hidden_dim {
            return Err(geometry::GeometryError::DimensionMismatch {
                expected: vocab_size * hidden_dim,
                got: data.len(),
            });
        }
        Ok(Self {
            vocab_size,
            hidden_dim,
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// Current schema version constant
// ---------------------------------------------------------------------------

/// Wire-format version 1 (original, no chaining).
pub const SCHEMA_VERSION: u16 = 1;

/// Wire-format version 2 (chained attestation with geometry drift).
pub const SCHEMA_VERSION_2: u16 = 2;

/// Wire-format version 3 (causal intervention scores).
pub const SCHEMA_VERSION_3: u16 = 3;

// ---------------------------------------------------------------------------
// SHA-256 helper (used by multiple crates, centralised here)
// ---------------------------------------------------------------------------

/// SHA-256 hash of arbitrary data. Convenience wrapper used by probe
/// readings, attestation hashing, etc.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ---------------------------------------------------------------------------
// Causal intervention score — result of a single causal_check call.
// Stored in the attestation so verifiers can inspect causality evidence.
// ---------------------------------------------------------------------------

/// Record of one causal intervention probe check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalScoreRecord {
    /// ‖output(h + δw) − output(h)‖₂
    pub delta_plus: f32,
    /// ‖output(h − δw) − output(h)‖₂
    pub delta_minus: f32,
    /// Causal consistency ∈ [-1, 1]. +1 = symmetric causal, 0 = one-sided.
    pub consistency: f32,
    /// consistency > threshold (default 0.5)
    pub is_causal: bool,
}

/// Per-probe directional drift record.
///
/// Measures how much the geometry changed specifically in the direction
/// a probe measures, not just globally (Frobenius norm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalDrift {
    /// Which probe this drift measurement is for.
    pub probe_name: String,
    /// |wᵀ(Φ_new − Φ_ref)w| / |wᵀΦ_ref w|
    pub drift: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multi-byte UTF-8 chars that happen to produce 64 *bytes* must not
    /// panic (they used to, because str slicing hit a non-char boundary).
    #[test]
    fn hex32_rejects_non_ascii_without_panic() {
        // 32 two-byte chars = 64 bytes total, but 32 chars (not 64).
        // serde_json will pass this as a 32-char String whose .len() != 64,
        // so the length check catches it. But if someone crafts exactly 64
        // bytes of multi-byte UTF-8, the ASCII-hex guard must fire.
        //
        // 64 bytes from 32 × U+00E9 ('é', 2-byte UTF-8):
        let bad: String = "é".repeat(32); // 32 chars, 64 bytes
        assert_eq!(bad.len(), 64, "test precondition");

        // Manually invoke the deserializer path via serde_json
        let json = format!(r#"{{"h":"{bad}"}}"#);
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(with = "hex32")]
            #[allow(dead_code)]
            h: [u8; 32],
        }
        let result: Result<W, _> = serde_json::from_str(&json);
        assert!(result.is_err(), "non-ASCII hex should be rejected");
    }

    #[test]
    fn hex64_rejects_non_ascii_without_panic() {
        let bad: String = "é".repeat(64); // 64 chars, 128 bytes
        assert_eq!(bad.len(), 128, "test precondition");

        let json = format!(r#"{{"s":"{bad}"}}"#);
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(with = "hex64")]
            #[allow(dead_code)]
            s: [u8; 64],
        }
        let result: Result<W, _> = serde_json::from_str(&json);
        assert!(result.is_err(), "non-ASCII hex should be rejected");
    }

    #[test]
    fn hex32_roundtrip() {
        let original: [u8; 32] = [0xDE; 32];
        let json = serde_json::to_string(&HexWrap32 { h: original }).unwrap();
        let decoded: HexWrap32 = serde_json::from_str(&json).unwrap();
        assert_eq!(original, decoded.h);
    }

    #[test]
    fn hex64_roundtrip() {
        let original: [u8; 64] = [0xAB; 64];
        let json = serde_json::to_string(&HexWrap64 { s: original }).unwrap();
        let decoded: HexWrap64 = serde_json::from_str(&json).unwrap();
        assert_eq!(original, decoded.s);
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct HexWrap32 {
        #[serde(with = "hex32")]
        h: [u8; 32],
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct HexWrap64 {
        #[serde(with = "hex64")]
        s: [u8; 64],
    }
}
