// ---------------------------------------------------------------------------
// Frame codec — length-prefixed binary framing for GOT/1.
//
//   Offset  Size   Field
//   0       4      Magic: "GOT1" (0x474F5431)
//   4       1      Message type (u8)
//   5       4      Payload length L (u32 BE)
//   9       L      Payload bytes
// ---------------------------------------------------------------------------

use crate::{WireError, MAGIC, MAX_PAYLOAD_SIZE};

/// GOT/1 message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    ExchangeReq = 0x01,
    ExchangeRsp = 0x02,
    VerifyReq = 0x03,
    VerifyRsp = 0x04,
    ChainReq = 0x05,
    ChainRsp = 0x06,
    BehavioralExchangeReq = 0x10,
    BehavioralExchangeRsp = 0x11,
    Error = 0xFF,
}

impl MessageType {
    pub fn from_byte(b: u8) -> Result<Self, WireError> {
        match b {
            0x01 => Ok(Self::ExchangeReq),
            0x02 => Ok(Self::ExchangeRsp),
            0x03 => Ok(Self::VerifyReq),
            0x04 => Ok(Self::VerifyRsp),
            0x05 => Ok(Self::ChainReq),
            0x06 => Ok(Self::ChainRsp),
            0x10 => Ok(Self::BehavioralExchangeReq),
            0x11 => Ok(Self::BehavioralExchangeRsp),
            0xFF => Ok(Self::Error),
            _ => Err(WireError::UnknownMessageType(b)),
        }
    }

    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// A decoded GOT/1 frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub message_type: MessageType,
    pub payload: Vec<u8>,
}

/// Frame header size: 4 (magic) + 1 (type) + 4 (length) = 9 bytes.
pub const FRAME_HEADER_SIZE: usize = 9;

impl Frame {
    /// Encode this frame into bytes.
    ///
    /// Returns `Err(WireError::PayloadTooLarge)` if the payload exceeds
    /// [`MAX_PAYLOAD_SIZE`].
    pub fn encode(&self) -> Result<Vec<u8>, WireError> {
        let len: u32 = self.payload.len().try_into().map_err(|_| {
            // payload.len() > u32::MAX
            WireError::PayloadTooLarge {
                size: u32::MAX,
                limit: MAX_PAYLOAD_SIZE,
            }
        })?;
        if len > MAX_PAYLOAD_SIZE {
            return Err(WireError::PayloadTooLarge {
                size: len,
                limit: MAX_PAYLOAD_SIZE,
            });
        }
        let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + self.payload.len());
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(self.message_type.to_byte());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        Ok(buf)
    }

    /// Decode a frame from bytes.  Returns the frame and the number of bytes consumed.
    pub fn decode(data: &[u8]) -> Result<(Self, usize), WireError> {
        if data.len() < FRAME_HEADER_SIZE {
            return Err(WireError::IncompleteFrame {
                needed: FRAME_HEADER_SIZE,
                got: data.len(),
            });
        }

        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != MAGIC {
            return Err(WireError::BadMagic(magic));
        }

        let message_type = MessageType::from_byte(data[4])?;
        let payload_len = u32::from_be_bytes([data[5], data[6], data[7], data[8]]);

        if payload_len > MAX_PAYLOAD_SIZE {
            return Err(WireError::PayloadTooLarge {
                size: payload_len,
                limit: MAX_PAYLOAD_SIZE,
            });
        }

        let total = FRAME_HEADER_SIZE + payload_len as usize;
        if data.len() < total {
            return Err(WireError::IncompleteFrame {
                needed: total,
                got: data.len(),
            });
        }

        let payload = data[FRAME_HEADER_SIZE..total].to_vec();
        Ok((
            Frame {
                message_type,
                payload,
            },
            total,
        ))
    }
}

/// Encode a GOT/1 ERROR frame.
pub fn encode_error(code: u32, message: &str) -> Frame {
    let msg_bytes = message.as_bytes();
    let mut payload = Vec::with_capacity(8 + msg_bytes.len());
    payload.extend_from_slice(&code.to_be_bytes());
    payload.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    payload.extend_from_slice(msg_bytes);
    Frame {
        message_type: MessageType::Error,
        payload,
    }
}

/// Decode a GOT/1 ERROR payload into (code, message).
pub fn decode_error(payload: &[u8]) -> Result<(u32, String), WireError> {
    if payload.len() < 8 {
        return Err(WireError::IncompleteFrame {
            needed: 8,
            got: payload.len(),
        });
    }
    let code = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let msg_len = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    if payload.len() < 8 + msg_len {
        return Err(WireError::IncompleteFrame {
            needed: 8 + msg_len,
            got: payload.len(),
        });
    }
    let message = String::from_utf8(payload[8..8 + msg_len].to_vec())
        .map_err(|e| WireError::Protocol(format!("invalid UTF-8 in error message: {e}")))?;
    Ok((code, message))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_all_types() {
        let types = [
            MessageType::ExchangeReq,
            MessageType::ExchangeRsp,
            MessageType::VerifyReq,
            MessageType::VerifyRsp,
            MessageType::ChainReq,
            MessageType::ChainRsp,
            MessageType::Error,
        ];

        for mt in types {
            let payload = vec![1, 2, 3, 4, 5];
            let frame = Frame {
                message_type: mt,
                payload: payload.clone(),
            };
            let encoded = frame.encode().unwrap();
            let (decoded, consumed) = Frame::decode(&encoded).unwrap();
            assert_eq!(consumed, encoded.len());
            assert_eq!(decoded.message_type, mt);
            assert_eq!(decoded.payload, payload);
        }
    }

    #[test]
    fn frame_empty_payload() {
        let frame = Frame {
            message_type: MessageType::ExchangeReq,
            payload: vec![],
        };
        let encoded = frame.encode().unwrap();
        assert_eq!(encoded.len(), FRAME_HEADER_SIZE);
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = vec![0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00];
        let err = Frame::decode(&data).unwrap_err();
        assert!(matches!(err, WireError::BadMagic(_)));

        // Correct magic but wrong byte
        data[0..4].copy_from_slice(&0x474F5432u32.to_be_bytes());
        let err = Frame::decode(&data).unwrap_err();
        assert!(matches!(err, WireError::BadMagic(0x474F5432)));
    }

    #[test]
    fn unknown_message_type_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(0x77); // unknown
        buf.extend_from_slice(&0u32.to_be_bytes());
        let err = Frame::decode(&buf).unwrap_err();
        assert!(matches!(err, WireError::UnknownMessageType(0x77)));
    }

    #[test]
    fn incomplete_frame_rejected() {
        let err = Frame::decode(&[0x47, 0x4F]).unwrap_err();
        assert!(matches!(err, WireError::IncompleteFrame { .. }));
    }

    #[test]
    fn payload_too_large_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(0x01);
        buf.extend_from_slice(&(MAX_PAYLOAD_SIZE + 1).to_be_bytes());
        let err = Frame::decode(&buf).unwrap_err();
        assert!(matches!(err, WireError::PayloadTooLarge { .. }));
    }

    #[test]
    fn error_frame_roundtrip() {
        let frame = encode_error(7, "envelope signature invalid");
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.message_type, MessageType::Error);
        let (code, msg) = decode_error(&decoded.payload).unwrap();
        assert_eq!(code, 7);
        assert_eq!(msg, "envelope signature invalid");
    }

    #[test]
    fn multiple_frames_in_buffer() {
        let f1 = Frame {
            message_type: MessageType::ExchangeReq,
            payload: vec![0xAA],
        };
        let f2 = Frame {
            message_type: MessageType::ExchangeRsp,
            payload: vec![0xBB, 0xCC],
        };
        let mut buf = f1.encode().unwrap();
        buf.extend_from_slice(&f2.encode().unwrap());

        let (d1, consumed1) = Frame::decode(&buf).unwrap();
        assert_eq!(d1.message_type, MessageType::ExchangeReq);
        assert_eq!(d1.payload, vec![0xAA]);

        let (d2, consumed2) = Frame::decode(&buf[consumed1..]).unwrap();
        assert_eq!(d2.message_type, MessageType::ExchangeRsp);
        assert_eq!(d2.payload, vec![0xBB, 0xCC]);
        assert_eq!(consumed1 + consumed2, buf.len());
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issue 38)
    // -----------------------------------------------------------------------

    /// Issue #38 (S-17): `decode_error()` must reject invalid UTF-8.
    #[test]
    fn sec_decode_error_rejects_invalid_utf8() {
        let code: u32 = 1;
        let bad_utf8: &[u8] = &[0x80, 0x81, 0xFF]; // not valid UTF-8
        let msg_len: u32 = bad_utf8.len() as u32;

        let mut payload = Vec::new();
        payload.extend_from_slice(&code.to_be_bytes());
        payload.extend_from_slice(&msg_len.to_be_bytes());
        payload.extend_from_slice(bad_utf8);

        let result = decode_error(&payload);
        assert!(matches!(result, Err(WireError::Protocol(_))), "decode_error must reject invalid UTF-8");
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issue 43 / N-1)
    // -----------------------------------------------------------------------

    /// Issue #43 (N-1): `Frame::encode()` must reject payloads exceeding
    /// `MAX_PAYLOAD_SIZE`.
    #[test]
    fn sec_frame_encode_rejects_oversized_payload() {
        let frame = Frame {
            message_type: MessageType::ExchangeReq,
            payload: vec![0u8; MAX_PAYLOAD_SIZE as usize + 1],
        };
        let err = frame.encode().unwrap_err();
        assert!(
            matches!(err, WireError::PayloadTooLarge { .. }),
            "encode() must reject payload > MAX_PAYLOAD_SIZE"
        );
    }
}
