// ---------------------------------------------------------------------------
// Wire codec for ExchangeRequest / ExchangeResponse.
//
// The protocol's typed exchange structures live in `got-wire::exchange` and
// have no built-in serialisation — the in-memory test path passes them by
// value through `perform_exchange`.  For network transport we need to lay
// them out as bytes; this module defines that layout.
//
// Format choice: a small fixed header + length-prefixed JSON for the
// `GeometricAttestation`s (which already implement serde Serialize /
// Deserialize) + the existing 200-byte canonical envelope encoding.  JSON
// is verbose but debuggable, and the bandwidth cost is negligible relative
// to a full attestation chain.  A production deployment that needs
// efficiency would swap this for bincode or a manually-laid-out binary
// format — the rest of the crate is agnostic.
//
//   ExchangeRequest layout:
//     [0..32]   agent_id          (32 bytes)
//     [32..232] envelope          (200 bytes; ExchangeEnvelope::to_bytes)
//     [232..236] chain_count      (u32 BE)
//     for i in 0..chain_count:
//       [.. .. + 4]  json_len     (u32 BE)
//       [.. .. + json_len] json   (UTF-8 JSON of GeometricAttestation)
//     [.. .. + 4]  current_len    (u32 BE)
//     [.. ..]      current_json   (UTF-8 JSON of GeometricAttestation)
//
//   ExchangeResponse layout:
//     [0]       verdict           (u8: 0x01 Accepted / 0x02 Rejected / 0x03 Error)
//     [1..5]    reason_len        (u32 BE)
//     [5..]     reason            (UTF-8 string)
//     followed by the same layout as ExchangeRequest.
//
// All `u32 BE` lengths are bounded by `MAX_MESSAGE_SIZE` so a malicious
// peer cannot dictate an unbounded allocation.
// ---------------------------------------------------------------------------

use got_core::GeometricAttestation;
use got_wire::envelope::{ExchangeEnvelope, ENVELOPE_SIZE};
use got_wire::exchange::{ExchangeRequest, ExchangeResponse, Verdict};

use crate::error::NetError;
use crate::transport::MAX_MESSAGE_SIZE;

// ---------------------------------------------------------------------------
// ExchangeRequest
// ---------------------------------------------------------------------------

pub fn encode_exchange_request(req: &ExchangeRequest) -> Result<Vec<u8>, NetError> {
    let mut buf = Vec::with_capacity(256 + req.chain.len() * 1024);
    buf.extend_from_slice(&req.agent_id);
    buf.extend_from_slice(&req.envelope.to_bytes());
    write_u32(&mut buf, req.chain.len() as u32);
    for att in &req.chain {
        write_attestation(&mut buf, att)?;
    }
    write_attestation(&mut buf, &req.current)?;
    if buf.len() > MAX_MESSAGE_SIZE {
        return Err(NetError::MessageTooLarge {
            size: buf.len(),
            limit: MAX_MESSAGE_SIZE,
        });
    }
    Ok(buf)
}

pub fn decode_exchange_request(data: &[u8]) -> Result<ExchangeRequest, NetError> {
    let mut cursor = Cursor::new(data);
    let agent_id = cursor.read_array::<32>()?;
    let envelope_bytes = cursor.read_array::<ENVELOPE_SIZE>()?;
    let envelope = ExchangeEnvelope::from_bytes(&envelope_bytes);
    let chain_count = cursor.read_u32()? as usize;
    if chain_count > MAX_MESSAGE_SIZE / 64 {
        // Sanity bound: a chain with more entries than fits in MAX_MESSAGE_SIZE
        // is impossible.  Reject early to avoid huge allocations.
        return Err(NetError::Codec(format!(
            "implausible chain length: {chain_count}"
        )));
    }
    let mut chain = Vec::with_capacity(chain_count);
    for _ in 0..chain_count {
        chain.push(read_attestation(&mut cursor)?);
    }
    let current = read_attestation(&mut cursor)?;
    cursor.finish()?;
    Ok(ExchangeRequest {
        agent_id,
        envelope,
        chain,
        current,
    })
}

// ---------------------------------------------------------------------------
// ExchangeResponse
// ---------------------------------------------------------------------------

pub fn encode_exchange_response(rsp: &ExchangeResponse) -> Result<Vec<u8>, NetError> {
    let mut buf = Vec::with_capacity(256 + rsp.chain.len() * 1024);
    buf.push(rsp.verdict.to_byte());
    let reason_bytes = rsp.reason.as_bytes();
    write_u32(&mut buf, reason_bytes.len() as u32);
    buf.extend_from_slice(reason_bytes);
    buf.extend_from_slice(&rsp.agent_id);
    buf.extend_from_slice(&rsp.envelope.to_bytes());
    write_u32(&mut buf, rsp.chain.len() as u32);
    for att in &rsp.chain {
        write_attestation(&mut buf, att)?;
    }
    write_attestation(&mut buf, &rsp.current)?;
    if buf.len() > MAX_MESSAGE_SIZE {
        return Err(NetError::MessageTooLarge {
            size: buf.len(),
            limit: MAX_MESSAGE_SIZE,
        });
    }
    Ok(buf)
}

pub fn decode_exchange_response(data: &[u8]) -> Result<ExchangeResponse, NetError> {
    let mut cursor = Cursor::new(data);
    let verdict_byte = cursor.read_u8()?;
    let verdict = Verdict::from_byte(verdict_byte)
        .map_err(|e| NetError::Codec(format!("verdict: {e}")))?;
    let reason_len = cursor.read_u32()? as usize;
    let reason_bytes = cursor.read_slice(reason_len)?;
    let reason = std::str::from_utf8(reason_bytes)
        .map_err(|e| NetError::Codec(format!("reason: {e}")))?
        .to_string();
    let agent_id = cursor.read_array::<32>()?;
    let envelope_bytes = cursor.read_array::<ENVELOPE_SIZE>()?;
    let envelope = ExchangeEnvelope::from_bytes(&envelope_bytes);
    let chain_count = cursor.read_u32()? as usize;
    if chain_count > MAX_MESSAGE_SIZE / 64 {
        return Err(NetError::Codec(format!(
            "implausible chain length: {chain_count}"
        )));
    }
    let mut chain = Vec::with_capacity(chain_count);
    for _ in 0..chain_count {
        chain.push(read_attestation(&mut cursor)?);
    }
    let current = read_attestation(&mut cursor)?;
    cursor.finish()?;
    Ok(ExchangeResponse {
        agent_id,
        envelope,
        verdict,
        chain,
        current,
        reason,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

fn write_attestation(buf: &mut Vec<u8>, att: &GeometricAttestation) -> Result<(), NetError> {
    let json = serde_json::to_vec(att)?;
    if json.len() > MAX_MESSAGE_SIZE {
        return Err(NetError::MessageTooLarge {
            size: json.len(),
            limit: MAX_MESSAGE_SIZE,
        });
    }
    write_u32(buf, json.len() as u32);
    buf.extend_from_slice(&json);
    Ok(())
}

fn read_attestation(cursor: &mut Cursor<'_>) -> Result<GeometricAttestation, NetError> {
    let len = cursor.read_u32()? as usize;
    if len > MAX_MESSAGE_SIZE {
        return Err(NetError::MessageTooLarge {
            size: len,
            limit: MAX_MESSAGE_SIZE,
        });
    }
    let json = cursor.read_slice(len)?;
    let att: GeometricAttestation = serde_json::from_slice(json)?;
    Ok(att)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8], NetError> {
        if self.remaining() < n {
            return Err(NetError::UnexpectedEof { expected: n });
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], NetError> {
        let s = self.read_slice(N)?;
        let mut arr = [0u8; N];
        arr.copy_from_slice(s);
        Ok(arr)
    }

    fn read_u8(&mut self) -> Result<u8, NetError> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    fn read_u32(&mut self) -> Result<u32, NetError> {
        let arr = self.read_array::<4>()?;
        Ok(u32::from_be_bytes(arr))
    }

    fn finish(self) -> Result<(), NetError> {
        if self.pos != self.data.len() {
            return Err(NetError::Codec(format!(
                "trailing bytes after decode: {} of {}",
                self.data.len() - self.pos,
                self.data.len()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use got_attest::assemble_and_sign;
    use got_core::{GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION};
    use got_wire::envelope::ExchangeEnvelope;
    use got_wire::exchange::Verdict;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[0xAA; 32])
    }

    fn make_attest(k: &SigningKey) -> GeometricAttestation {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let a = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "codec-test".into(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0x22; 32],
            timestamp: now,
            corpus_version: "c1".into(),
            probe_version: "p1".into(),
            layer_readings: vec![vec![1.0, 2.0]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: None,
            geometry_hash: None,
            geometry_drift: None,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };
        assemble_and_sign(a, k).unwrap()
    }

    fn make_envelope(k: &SigningKey, att: &GeometricAttestation) -> ExchangeEnvelope {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        ExchangeEnvelope::create([0x42; 32], [0x77; 32], att, None, now, k).unwrap()
    }

    #[test]
    fn exchange_request_roundtrip() {
        let k = key();
        let att = make_attest(&k);
        let env = make_envelope(&k, &att);
        let req = ExchangeRequest {
            agent_id: [0xAB; 32],
            envelope: env,
            chain: vec![],
            current: att,
        };
        let bytes = encode_exchange_request(&req).unwrap();
        let decoded = decode_exchange_request(&bytes).unwrap();
        assert_eq!(decoded.agent_id, req.agent_id);
        assert_eq!(decoded.envelope.to_bytes(), req.envelope.to_bytes());
        assert_eq!(decoded.chain.len(), 0);
        assert_eq!(decoded.current.signature, req.current.signature);
    }

    #[test]
    fn exchange_request_with_chain_roundtrip() {
        let k = key();
        let anchor = make_attest(&k);
        let current = make_attest(&k);
        let env = make_envelope(&k, &current);
        let req = ExchangeRequest {
            agent_id: [0xAB; 32],
            envelope: env,
            chain: vec![anchor.clone()],
            current: current.clone(),
        };
        let bytes = encode_exchange_request(&req).unwrap();
        let decoded = decode_exchange_request(&bytes).unwrap();
        assert_eq!(decoded.chain.len(), 1);
        assert_eq!(decoded.chain[0].signature, anchor.signature);
        assert_eq!(decoded.current.signature, current.signature);
    }

    #[test]
    fn exchange_response_roundtrip() {
        let k = key();
        let att = make_attest(&k);
        let env = make_envelope(&k, &att);
        let rsp = ExchangeResponse {
            agent_id: [0xCD; 32],
            envelope: env,
            verdict: Verdict::Accepted,
            chain: vec![],
            current: att,
            reason: "ok".into(),
        };
        let bytes = encode_exchange_response(&rsp).unwrap();
        let decoded = decode_exchange_response(&bytes).unwrap();
        assert_eq!(decoded.verdict, Verdict::Accepted);
        assert_eq!(decoded.reason, "ok");
        assert_eq!(decoded.agent_id, rsp.agent_id);
    }

    #[test]
    fn rejects_trailing_bytes() {
        let k = key();
        let att = make_attest(&k);
        let env = make_envelope(&k, &att);
        let req = ExchangeRequest {
            agent_id: [0xAB; 32],
            envelope: env,
            chain: vec![],
            current: att,
        };
        let mut bytes = encode_exchange_request(&req).unwrap();
        bytes.push(0x00);
        let err = decode_exchange_request(&bytes).unwrap_err();
        assert!(matches!(err, NetError::Codec(_)));
    }

    #[test]
    fn rejects_truncated_input() {
        let k = key();
        let att = make_attest(&k);
        let env = make_envelope(&k, &att);
        let req = ExchangeRequest {
            agent_id: [0xAB; 32],
            envelope: env,
            chain: vec![],
            current: att,
        };
        let bytes = encode_exchange_request(&req).unwrap();
        let truncated = &bytes[..bytes.len() - 10];
        let err = decode_exchange_request(truncated).unwrap_err();
        assert!(matches!(err, NetError::UnexpectedEof { .. } | NetError::Codec(_)));
    }
}
