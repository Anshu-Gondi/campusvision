/// scheduler.rs
///
/// Two-lane priority scheduler:
///   - DIRECT lane: user-triggered verifications (2-3 per session), fast semaphore, never queued behind CCTV
///   - CCTV lane:   continuous background frames, runs on remaining capacity
///
/// When a DIRECT permit is requested it always gets the next free slot.
/// CCTV workers hold a weak permit that is pre-emptible: they check
/// `direct_waiting` before starting each unit of work and yield if set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Semaphore, SemaphorePermit, OwnedSemaphorePermit};

/// Tuning knobs — adjust to your instance size.
/// On a 4-vCPU cloud node a good split is 3 direct + 5 CCTV = 8 total
/// (inference workers are separate threads, this controls *request* concurrency).
const DIRECT_SLOTS: usize = 3;
const CCTV_SLOTS: usize = 5;

#[derive(Clone)]
pub struct PriorityScheduler {
    /// Fast lane — only direct verification calls acquire these.
    direct_sem: Arc<Semaphore>,
    /// Background lane — CCTV frames acquire these.
    cctv_sem: Arc<Semaphore>,
    /// How many direct callers are currently waiting. CCTV workers
    /// read this before each sub-step and back off if > 0.
    direct_waiting: Arc<AtomicUsize>,
}

/// A permit that auto-releases when dropped.
pub struct DirectPermit {
    _inner: OwnedSemaphorePermit,
    waiting: Arc<AtomicUsize>,
}

impl Drop for DirectPermit {
    fn drop(&mut self) {
        // Permit is released by OwnedSemaphorePermit drop.
        // Nothing extra needed — waiting counter was decremented when acquired.
        let _ = &self.waiting; // keep Arc alive
    }
}

/// A permit for CCTV work. Holds the semaphore permit AND exposes
/// a method to check if a direct call is waiting (so the holder can
/// yield voluntarily between pipeline stages).
pub struct CctvPermit {
    _inner: OwnedSemaphorePermit,
    direct_waiting: Arc<AtomicUsize>,
}

impl CctvPermit {
    /// Returns true if a direct call is currently waiting for a slot.
    /// CCTV workers should call this between expensive stages (after
    /// detection, before search) and drop the frame if true — the frame
    /// will be retried on the next camera tick anyway.
    #[inline]
    pub fn should_yield(&self) -> bool {
        self.direct_waiting.load(Ordering::Acquire) > 0
    }
}

impl PriorityScheduler {
    pub fn new() -> Self {
        Self {
            direct_sem: Arc::new(Semaphore::new(DIRECT_SLOTS)),
            cctv_sem: Arc::new(Semaphore::new(CCTV_SLOTS)),
            direct_waiting: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Acquire a direct permit. Increments `direct_waiting` immediately
    /// so in-flight CCTV workers see it before their next stage.
    /// Awaiting this is cancel-safe.
    pub async fn acquire_direct(&self) -> DirectPermit {
        // Signal intent *before* waiting so CCTV can back off.
        self.direct_waiting.fetch_add(1, Ordering::Release);

        let permit = Arc::clone(&self.direct_sem)
            .acquire_owned()
            .await
            .expect("direct semaphore closed");

        // We have the slot — decrement waiting.
        self.direct_waiting.fetch_sub(1, Ordering::Release);

        DirectPermit {
            _inner: permit,
            waiting: Arc::clone(&self.direct_waiting),
        }
    }

    /// Acquire a CCTV permit. Returns None if the queue is full
    /// (non-blocking try) — callers should drop the frame rather than
    /// block, since the next camera tick will produce a fresh frame.
    pub fn try_acquire_cctv(&self) -> Option<CctvPermit> {
        Arc::clone(&self.cctv_sem)
            .try_acquire_owned()
            .ok()
            .map(|p| CctvPermit {
                _inner: p,
                direct_waiting: Arc::clone(&self.direct_waiting),
            })
    }

    /// Blocking CCTV acquire — use for the frame batcher task which
    /// should wait rather than drop when all slots are busy.
    pub async fn acquire_cctv(&self) -> CctvPermit {
        let permit = Arc::clone(&self.cctv_sem)
            .acquire_owned()
            .await
            .expect("cctv semaphore closed");

        CctvPermit {
            _inner: permit,
            direct_waiting: Arc::clone(&self.direct_waiting),
        }
    }

    pub fn direct_available(&self) -> usize {
        self.direct_sem.available_permits()
    }

    pub fn cctv_available(&self) -> usize {
        self.cctv_sem.available_permits()
    }
}