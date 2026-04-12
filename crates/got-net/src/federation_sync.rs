// ---------------------------------------------------------------------------
// FederationSyncManager — async polling loop for federation member refresh.
//
// Sits above the `got_wire::federation::FederationSyncSource` trait,
// driving each registered source on its own schedule.  Implementations
// of `FederationSyncSource` are sync (the trait is sync, the
// implementations may block on I/O), so each poll runs inside
// `tokio::task::spawn_blocking` to avoid stalling the async runtime
// on a slow file read or socket connect.
//
// What this manager does NOT do:
//   - It does not load registries, parse TOML, or build
//     `FederatedRegistry` instances.  Its job is to fetch the freshest
//     bytes for each member and surface them to the caller via
//     `latest_snapshot(name)`; the caller is responsible for the
//     parsing-and-rebuilding step (typically: when a snapshot
//     changes, call `TrustRegistry::load(path, &digest)` against the
//     new bytes and reconstruct a `FederatedRegistry`).  Keeping
//     parsing out of the manager makes the failure modes cleaner — a
//     fetch can succeed but a parse can fail, and the caller knows
//     which is which.
//   - It does not arbitrate between members.  Each member is polled
//     independently; there is no cross-member coordination.
//   - It does not push updates.  Polling is the only refresh
//     strategy here; sources that want push semantics can implement
//     a `FederationSyncSource` that internally pushes new content
//     into a buffer that `fetch()` drains.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use got_wire::federation::{FederationSyncSource, SyncedRegistry};

/// Per-source refresh policy.
#[derive(Debug, Clone, Copy)]
pub struct RefreshPolicy {
    /// Base interval between successful fetches.
    pub interval: Duration,
    /// Initial backoff after a failed fetch.  Doubles on each
    /// consecutive failure up to `max_backoff`.
    pub initial_backoff: Duration,
    /// Maximum backoff between failed retries.
    pub max_backoff: Duration,
    /// If a source has not been successfully refreshed within this
    /// duration, the manager marks it `Stale` in the status report.
    /// The verifier should refuse new exchanges from members whose
    /// status is `Stale`.
    pub max_staleness: Duration,
}

impl RefreshPolicy {
    /// Sensible defaults for a low-traffic federation: refresh every
    /// hour, exponential backoff from 1 minute up to 1 hour, mark
    /// stale after 24 hours of failure.
    pub fn defaults() -> Self {
        Self {
            interval: Duration::from_secs(3600),
            initial_backoff: Duration::from_secs(60),
            max_backoff: Duration::from_secs(3600),
            max_staleness: Duration::from_secs(86_400),
        }
    }

    /// Stricter defaults for critical infrastructure: refresh every
    /// 5 minutes, mark stale after 1 hour.
    pub fn critical() -> Self {
        Self {
            interval: Duration::from_secs(300),
            initial_backoff: Duration::from_secs(30),
            max_backoff: Duration::from_secs(600),
            max_staleness: Duration::from_secs(3600),
        }
    }
}

/// Status of a single sync source as observed by the manager.
#[derive(Debug, Clone)]
pub struct SyncStatus {
    pub name: String,
    pub last_attempt: Option<u64>,
    pub last_success: Option<u64>,
    pub last_error: Option<String>,
    pub consecutive_failures: u32,
    pub stale: bool,
    pub current_digest: Option<[u8; 32]>,
}

/// Internal per-source state.
#[derive(Debug)]
struct SourceState {
    source: Arc<dyn FederationSyncSource>,
    policy: RefreshPolicy,
    status: SyncStatus,
    snapshot: Option<SyncedRegistry>,
}

/// Async manager that supervises polling for a set of named
/// `FederationSyncSource`s.
///
/// Construct with `new()`, register sources with `register()`, then
/// call `spawn()` to start the polling loop.  The returned
/// `JoinHandle` is the task supervisor — `abort()` it to stop
/// polling.  Snapshots and statuses are accessible via
/// `latest_snapshot()` and `status()`.
#[derive(Debug)]
pub struct FederationSyncManager {
    inner: Arc<RwLock<HashMap<String, SourceState>>>,
}

impl Default for FederationSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FederationSyncManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new sync source.  Returns an error only if a
    /// source with the same name is already registered.
    pub async fn register(
        &self,
        name: impl Into<String>,
        source: Arc<dyn FederationSyncSource>,
        policy: RefreshPolicy,
    ) -> Result<(), String> {
        let name = name.into();
        let mut guard = self.inner.write().await;
        if guard.contains_key(&name) {
            return Err(format!("source {name:?} already registered"));
        }
        guard.insert(
            name.clone(),
            SourceState {
                source,
                policy,
                status: SyncStatus {
                    name,
                    last_attempt: None,
                    last_success: None,
                    last_error: None,
                    consecutive_failures: 0,
                    stale: false,
                    current_digest: None,
                },
                snapshot: None,
            },
        );
        Ok(())
    }

    /// Latest snapshot for `name`, or `None` if the source has not
    /// yet successfully fetched once.
    pub async fn latest_snapshot(&self, name: &str) -> Option<SyncedRegistry> {
        let guard = self.inner.read().await;
        guard.get(name).and_then(|s| s.snapshot.clone())
    }

    /// Current status for `name`.
    pub async fn status(&self, name: &str) -> Option<SyncStatus> {
        let guard = self.inner.read().await;
        guard.get(name).map(|s| s.status.clone())
    }

    /// Statuses for every registered source.
    pub async fn all_statuses(&self) -> Vec<SyncStatus> {
        let guard = self.inner.read().await;
        guard.values().map(|s| s.status.clone()).collect()
    }

    /// Run one synchronous refresh of `name` immediately, blocking
    /// the caller until the fetch completes (or fails).  Useful for
    /// the initial bootstrap before the polling loop has had time to
    /// run, and for tests that want deterministic refresh timing.
    pub async fn refresh_once(&self, name: &str) -> Result<(), String> {
        let (source, since) = {
            let guard = self.inner.read().await;
            let state = guard.get(name).ok_or_else(|| format!("unknown source {name:?}"))?;
            (state.source.clone(), state.snapshot.as_ref().map(|s| s.digest))
        };

        // Run the sync fetch on a blocking thread so a slow source
        // does not stall the async caller.
        let fetch_result =
            tokio::task::spawn_blocking(move || source.fetch(since))
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?;

        let now = unix_now();
        let mut guard = self.inner.write().await;
        let state = guard.get_mut(name).ok_or_else(|| format!("unknown source {name:?}"))?;
        state.status.last_attempt = Some(now);

        match fetch_result {
            Ok(Some(snapshot)) => {
                state.status.last_success = Some(now);
                state.status.last_error = None;
                state.status.consecutive_failures = 0;
                state.status.stale = false;
                state.status.current_digest = Some(snapshot.digest);
                state.snapshot = Some(snapshot);
                Ok(())
            }
            Ok(None) => {
                // No change is also a successful refresh.
                state.status.last_success = Some(now);
                state.status.last_error = None;
                state.status.consecutive_failures = 0;
                state.status.stale = false;
                Ok(())
            }
            Err(e) => {
                let msg = e.to_string();
                state.status.last_error = Some(msg.clone());
                state.status.consecutive_failures =
                    state.status.consecutive_failures.saturating_add(1);
                // Mark stale if we've blown past max_staleness since
                // the last success (or, if there was never a success,
                // since registration — approximated as "right now").
                if let Some(last_success) = state.status.last_success {
                    state.status.stale = (now - last_success)
                        > state.policy.max_staleness.as_secs();
                } else {
                    state.status.stale = true;
                }
                Err(msg)
            }
        }
    }

    /// Spawn a tokio task that polls every registered source on its
    /// own schedule.  Returns a `JoinHandle` that the caller can
    /// `abort()` to stop polling.
    ///
    /// The task loops over all registered sources every `tick`
    /// (default 1 second), waking each source when its individual
    /// `interval` (or its current `backoff`) has elapsed since the
    /// last attempt.  Sources fetched from disk usually return
    /// almost immediately; HTTP sources are bounded by the source
    /// implementation's own timeouts.
    pub fn spawn(self: Arc<Self>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let tick = Duration::from_secs(1);
            loop {
                let names: Vec<String> = {
                    let guard = self.inner.read().await;
                    guard.keys().cloned().collect()
                };
                for name in names {
                    let now = unix_now();
                    let should_attempt = {
                        let guard = self.inner.read().await;
                        let state = match guard.get(&name) {
                            Some(s) => s,
                            None => continue,
                        };
                        next_attempt_due(&state.status, &state.policy, now)
                    };
                    if should_attempt {
                        // refresh_once handles all status updates.
                        let _ = self.refresh_once(&name).await;
                    }
                }
                tokio::time::sleep(tick).await;
            }
        })
    }
}

/// Decide whether `source` is due for a fetch attempt at `now_unix`.
///
/// First attempt: always due (last_attempt = None).
/// Subsequent successful attempts: due after `policy.interval`.
/// After failures: due after exponential backoff bounded by
/// `policy.max_backoff`.
fn next_attempt_due(status: &SyncStatus, policy: &RefreshPolicy, now_unix: u64) -> bool {
    let last = match status.last_attempt {
        Some(t) => t,
        None => return true,
    };
    let elapsed = now_unix.saturating_sub(last);
    let wait = if status.consecutive_failures == 0 {
        policy.interval.as_secs()
    } else {
        // 2^(failures-1) * initial_backoff, capped at max_backoff.
        let factor = 1u64.checked_shl(status.consecutive_failures.min(20) - 1).unwrap_or(1);
        let proposed = policy.initial_backoff.as_secs().saturating_mul(factor);
        proposed.min(policy.max_backoff.as_secs())
    };
    elapsed >= wait
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use got_wire::federation::{FileSyncSource, StaticSyncSource};

    #[tokio::test]
    async fn register_and_refresh_static_source() {
        let mgr = FederationSyncManager::new();
        let src = Arc::new(StaticSyncSource::new("eu", b"hello".to_vec(), 1_000_000))
            as Arc<dyn FederationSyncSource>;
        mgr.register("eu", src, RefreshPolicy::defaults()).await.unwrap();

        // First refresh should populate the snapshot.
        mgr.refresh_once("eu").await.unwrap();
        let snap = mgr.latest_snapshot("eu").await.expect("snapshot");
        assert_eq!(snap.bytes, b"hello");

        let status = mgr.status("eu").await.unwrap();
        assert!(status.last_success.is_some());
        assert_eq!(status.consecutive_failures, 0);
        assert!(!status.stale);
    }

    #[tokio::test]
    async fn refresh_with_no_change_keeps_status_clean() {
        let mgr = FederationSyncManager::new();
        let src = Arc::new(StaticSyncSource::new("eu", b"hello".to_vec(), 1_000_000))
            as Arc<dyn FederationSyncSource>;
        mgr.register("eu", src, RefreshPolicy::defaults()).await.unwrap();
        mgr.refresh_once("eu").await.unwrap();
        // Second refresh: source returns None (digest matches), but
        // the manager treats it as a successful refresh.
        mgr.refresh_once("eu").await.unwrap();
        let status = mgr.status("eu").await.unwrap();
        assert_eq!(status.consecutive_failures, 0);
        assert!(!status.stale);
    }

    #[tokio::test]
    async fn refresh_failure_increments_failure_counter() {
        let mgr = FederationSyncManager::new();
        let src = Arc::new(FileSyncSource::new("eu", "/this/does/not/exist.toml"))
            as Arc<dyn FederationSyncSource>;
        mgr.register("eu", src, RefreshPolicy::defaults()).await.unwrap();
        let err = mgr.refresh_once("eu").await;
        assert!(err.is_err());
        let status = mgr.status("eu").await.unwrap();
        assert_eq!(status.consecutive_failures, 1);
        assert!(status.last_error.is_some());
        // Source has never succeeded, so it is immediately stale.
        assert!(status.stale);
    }

    #[tokio::test]
    async fn file_source_picks_up_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.toml");
        std::fs::write(&path, b"v=1").unwrap();

        let mgr = FederationSyncManager::new();
        let src = Arc::new(FileSyncSource::new("eu", &path)) as Arc<dyn FederationSyncSource>;
        mgr.register("eu", src, RefreshPolicy::defaults()).await.unwrap();
        mgr.refresh_once("eu").await.unwrap();
        let first = mgr.latest_snapshot("eu").await.unwrap();
        assert_eq!(first.bytes, b"v=1");

        // Modify the file and refresh.
        std::fs::write(&path, b"v=2").unwrap();
        mgr.refresh_once("eu").await.unwrap();
        let second = mgr.latest_snapshot("eu").await.unwrap();
        assert_eq!(second.bytes, b"v=2");
        assert_ne!(second.digest, first.digest);
    }

    #[tokio::test]
    async fn cannot_register_duplicate_name() {
        let mgr = FederationSyncManager::new();
        let src = Arc::new(StaticSyncSource::new("eu", vec![], 0))
            as Arc<dyn FederationSyncSource>;
        mgr.register("eu", src.clone(), RefreshPolicy::defaults()).await.unwrap();
        let err = mgr.register("eu", src, RefreshPolicy::defaults()).await;
        assert!(err.is_err());
    }

    #[test]
    fn next_attempt_due_first_call_always_true() {
        let status = SyncStatus {
            name: "eu".into(),
            last_attempt: None,
            last_success: None,
            last_error: None,
            consecutive_failures: 0,
            stale: false,
            current_digest: None,
        };
        assert!(next_attempt_due(&status, &RefreshPolicy::defaults(), 0));
    }

    #[test]
    fn next_attempt_due_respects_interval_after_success() {
        let status = SyncStatus {
            name: "eu".into(),
            last_attempt: Some(1_000_000),
            last_success: Some(1_000_000),
            last_error: None,
            consecutive_failures: 0,
            stale: false,
            current_digest: None,
        };
        let policy = RefreshPolicy {
            interval: Duration::from_secs(60),
            ..RefreshPolicy::defaults()
        };
        // Just after the success — not due yet.
        assert!(!next_attempt_due(&status, &policy, 1_000_030));
        // Past the interval — due.
        assert!(next_attempt_due(&status, &policy, 1_000_061));
    }

    #[test]
    fn next_attempt_due_exponential_backoff_after_failure() {
        let policy = RefreshPolicy {
            initial_backoff: Duration::from_secs(60),
            max_backoff: Duration::from_secs(3600),
            ..RefreshPolicy::defaults()
        };
        // 1st failure: backoff = 60s
        let status = SyncStatus {
            name: "eu".into(),
            last_attempt: Some(0),
            last_success: None,
            last_error: Some("oops".into()),
            consecutive_failures: 1,
            stale: false,
            current_digest: None,
        };
        assert!(!next_attempt_due(&status, &policy, 30));
        assert!(next_attempt_due(&status, &policy, 61));

        // 4th failure: backoff = 60 * 2^3 = 480s
        let status = SyncStatus {
            consecutive_failures: 4,
            ..status
        };
        assert!(!next_attempt_due(&status, &policy, 400));
        assert!(next_attempt_due(&status, &policy, 481));

        // 20th failure: capped at max_backoff = 3600s
        let status = SyncStatus {
            consecutive_failures: 20,
            ..status
        };
        assert!(!next_attempt_due(&status, &policy, 3000));
        assert!(next_attempt_due(&status, &policy, 3601));
    }
}
