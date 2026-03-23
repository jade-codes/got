// ---------------------------------------------------------------------------
// ValueSpaceStore — persistence for BehavioralValueSpace snapshots.
//
// Follows the same trait-based pattern as got-store::AttestationStore.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::Path;

use crate::attestation::BehavioralAttestation;
use crate::value_space::BehavioralValueSpace;
use crate::ProxyError;

/// Content-addressed store ID: SHA-256 of the value space hash.
pub type ValueSpaceId = [u8; 32];

/// Abstract value space store.
///
/// Implementations must enforce:
/// - Append-only: no mutation or deletion.
/// - Idempotent duplicate inserts.
pub trait ValueSpaceStore {
    /// Store a value space snapshot. Returns its content-addressed ID.
    fn store_snapshot(&mut self, space: &BehavioralValueSpace) -> Result<ValueSpaceId, ProxyError>;

    /// Retrieve a value space snapshot by content hash.
    fn get_snapshot(&self, id: &ValueSpaceId) -> Option<&BehavioralValueSpace>;

    /// Store a behavioral attestation.
    fn store_attestation(&mut self, attestation: &BehavioralAttestation)
        -> Result<[u8; 32], ProxyError>;

    /// Retrieve attestations for a target model (ordered by sequence number).
    fn attestations_for_model(&self, model_id: &str) -> Vec<&BehavioralAttestation>;

    /// Total number of snapshots.
    fn snapshot_count(&self) -> usize;

    /// Total number of attestations.
    fn attestation_count(&self) -> usize;
}

/// In-memory implementation of ValueSpaceStore.
#[derive(Debug, Default)]
pub struct MemoryValueSpaceStore {
    snapshots: HashMap<ValueSpaceId, BehavioralValueSpace>,
    attestations: Vec<BehavioralAttestation>,
    attestation_index: HashMap<[u8; 32], usize>,
}

impl MemoryValueSpaceStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ValueSpaceStore for MemoryValueSpaceStore {
    fn store_snapshot(&mut self, space: &BehavioralValueSpace) -> Result<ValueSpaceId, ProxyError> {
        let id = space.hash();
        self.snapshots.entry(id).or_insert_with(|| space.clone());
        Ok(id)
    }

    fn get_snapshot(&self, id: &ValueSpaceId) -> Option<&BehavioralValueSpace> {
        self.snapshots.get(id)
    }

    fn store_attestation(
        &mut self,
        attestation: &BehavioralAttestation,
    ) -> Result<[u8; 32], ProxyError> {
        let hash = crate::attestation::attestation_hash(attestation);
        if self.attestation_index.contains_key(&hash) {
            return Ok(hash); // idempotent
        }
        let idx = self.attestations.len();
        self.attestations.push(attestation.clone());
        self.attestation_index.insert(hash, idx);
        Ok(hash)
    }

    fn attestations_for_model(&self, model_id: &str) -> Vec<&BehavioralAttestation> {
        let mut results: Vec<&BehavioralAttestation> = self
            .attestations
            .iter()
            .filter(|a| a.target_model_id == model_id)
            .collect();
        results.sort_by_key(|a| a.sequence_number);
        results
    }

    fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    fn attestation_count(&self) -> usize {
        self.attestations.len()
    }
}

/// File-backed implementation of ValueSpaceStore.
///
/// Stores snapshots and attestations as JSON files in a directory.
/// Each file is named by its hex-encoded content hash.
pub struct FileValueSpaceStore {
    dir: std::path::PathBuf,
    // In-memory cache for reads
    snapshots: HashMap<ValueSpaceId, BehavioralValueSpace>,
    attestations: Vec<BehavioralAttestation>,
    attestation_index: HashMap<[u8; 32], usize>,
}

impl FileValueSpaceStore {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self, ProxyError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(dir.join("snapshots"))
            .map_err(|e| ProxyError::Io(e.to_string()))?;
        std::fs::create_dir_all(dir.join("attestations"))
            .map_err(|e| ProxyError::Io(e.to_string()))?;

        let mut store = Self {
            dir,
            snapshots: HashMap::new(),
            attestations: Vec::new(),
            attestation_index: HashMap::new(),
        };
        store.load_existing()?;
        Ok(store)
    }

    fn load_existing(&mut self) -> Result<(), ProxyError> {
        // Load snapshots
        let snap_dir = self.dir.join("snapshots");
        if snap_dir.exists() {
            for entry in std::fs::read_dir(&snap_dir).map_err(|e| ProxyError::Io(e.to_string()))? {
                let entry = entry.map_err(|e| ProxyError::Io(e.to_string()))?;
                let data =
                    std::fs::read_to_string(entry.path()).map_err(|e| ProxyError::Io(e.to_string()))?;
                if let Ok(space) = serde_json::from_str::<BehavioralValueSpace>(&data) {
                    let id = space.hash();
                    self.snapshots.insert(id, space);
                }
            }
        }

        // Load attestations
        let att_dir = self.dir.join("attestations");
        if att_dir.exists() {
            for entry in std::fs::read_dir(&att_dir).map_err(|e| ProxyError::Io(e.to_string()))? {
                let entry = entry.map_err(|e| ProxyError::Io(e.to_string()))?;
                let data =
                    std::fs::read_to_string(entry.path()).map_err(|e| ProxyError::Io(e.to_string()))?;
                if let Ok(att) = serde_json::from_str::<BehavioralAttestation>(&data) {
                    let hash = crate::attestation::attestation_hash(&att);
                    if !self.attestation_index.contains_key(&hash) {
                        let idx = self.attestations.len();
                        self.attestations.push(att);
                        self.attestation_index.insert(hash, idx);
                    }
                }
            }
        }

        Ok(())
    }

    fn hex_name(hash: &[u8; 32]) -> String {
        hash.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn write_atomic(&self, subdir: &str, hash: &[u8; 32], data: &str) -> Result<(), ProxyError> {
        let path = self.dir.join(subdir).join(format!("{}.json", Self::hex_name(hash)));
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, data).map_err(|e| ProxyError::Io(e.to_string()))?;
        std::fs::rename(&tmp_path, &path).map_err(|e| ProxyError::Io(e.to_string()))?;
        Ok(())
    }
}

impl ValueSpaceStore for FileValueSpaceStore {
    fn store_snapshot(&mut self, space: &BehavioralValueSpace) -> Result<ValueSpaceId, ProxyError> {
        let id = space.hash();
        if self.snapshots.contains_key(&id) {
            return Ok(id);
        }
        let json = serde_json::to_string_pretty(space)
            .map_err(|e| ProxyError::Serialisation(e.to_string()))?;
        self.write_atomic("snapshots", &id, &json)?;
        self.snapshots.insert(id, space.clone());
        Ok(id)
    }

    fn get_snapshot(&self, id: &ValueSpaceId) -> Option<&BehavioralValueSpace> {
        self.snapshots.get(id)
    }

    fn store_attestation(
        &mut self,
        attestation: &BehavioralAttestation,
    ) -> Result<[u8; 32], ProxyError> {
        let hash = crate::attestation::attestation_hash(attestation);
        if self.attestation_index.contains_key(&hash) {
            return Ok(hash);
        }
        let json = serde_json::to_string_pretty(attestation)
            .map_err(|e| ProxyError::Serialisation(e.to_string()))?;
        self.write_atomic("attestations", &hash, &json)?;
        let idx = self.attestations.len();
        self.attestations.push(attestation.clone());
        self.attestation_index.insert(hash, idx);
        Ok(hash)
    }

    fn attestations_for_model(&self, model_id: &str) -> Vec<&BehavioralAttestation> {
        let mut results: Vec<&BehavioralAttestation> = self
            .attestations
            .iter()
            .filter(|a| a.target_model_id == model_id)
            .collect();
        results.sort_by_key(|a| a.sequence_number);
        results
    }

    fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    fn attestation_count(&self) -> usize {
        self.attestations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_roundtrip() {
        let mut store = MemoryValueSpaceStore::new();
        let space = BehavioralValueSpace::new([0xAA; 32]);
        let id = store.store_snapshot(&space).unwrap();
        let retrieved = store.get_snapshot(&id).unwrap();
        assert_eq!(retrieved.reference_geometry_hash, space.reference_geometry_hash);
    }

    #[test]
    fn memory_store_idempotent() {
        let mut store = MemoryValueSpaceStore::new();
        let space = BehavioralValueSpace::new([0xAA; 32]);
        let id1 = store.store_snapshot(&space).unwrap();
        let id2 = store.store_snapshot(&space).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.snapshot_count(), 1);
    }
}
