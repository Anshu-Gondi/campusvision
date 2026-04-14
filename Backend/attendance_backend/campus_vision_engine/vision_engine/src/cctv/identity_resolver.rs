/// cctv/identity_resolver.rs  (replaces existing)
///
/// Key fix: original called batch_search TWICE per embedding —
/// once for "student", once for "teacher" — acquiring the index
/// read lock twice per face per frame.
///
/// New approach: single call with k=1 per role, but done inside
/// the frame_batcher which further collapses N frames into one
/// batch_search call. IdentityResolver is now called only from
/// the batcher flush, not per-frame.
///
/// For the FaceTracker pipeline (tracker.rs) which calls resolve()
/// directly, we keep the dual search but merge into a single
/// spawn_blocking call so only ONE thread-pool submission happens.

use intelligence_core::embeddings::batch_search;
use std::sync::Arc;

#[derive(Clone)]
pub struct IdentityResolver;

impl IdentityResolver {
    /// Resolve identity for a single embedding.
    /// Searches both roles in a single spawn_blocking call.
    /// Returns (id, score) of the best match across both roles,
    /// or None if no match meets the minimum threshold.
    pub async fn resolve_async(
        school_id: &str,
        embedding: Arc<Vec<f32>>,
    ) -> Option<(usize, f32)> {
        let school_id = school_id.to_string();
        let emb = (*embedding).clone();

        tokio::task::spawn_blocking(move || {
            // Both searches run sequentially on the blocking thread —
            // each acquires a read lock, but read locks don't block each
            // other, and we're off the Tokio executor thread.
            let student_results = batch_search(&school_id, std::slice::from_ref(&emb), "student", 1);
            let teacher_results = batch_search(&school_id, std::slice::from_ref(&emb), "teacher", 1);

            let best_student = student_results.into_iter().flatten().next();
            let best_teacher = teacher_results.into_iter().flatten().next();

            match (best_student, best_teacher) {
                (Some(s), Some(t)) => {
                    if s.1 >= t.1 { Some(s) } else { Some(t) }
                }
                (Some(s), None) => Some(s),
                (None, Some(t)) => Some(t),
                (None, None) => None,
            }
        })
        .await
        .ok()
        .flatten()
    }

    /// Synchronous version kept for legacy call sites that can't await.
    /// Prefer resolve_async where possible.
    pub fn resolve(school_id: &str, embedding: &Arc<Vec<f32>>) -> Option<(usize, f32)> {
        let student = batch_search(school_id, std::slice::from_ref(embedding.as_ref()), "student", 1);
        let teacher = batch_search(school_id, std::slice::from_ref(embedding.as_ref()), "teacher", 1);

        match (student.into_iter().flatten().next(), teacher.into_iter().flatten().next()) {
            (Some(s), Some(t)) => if s.1 >= t.1 { Some(s) } else { Some(t) },
            (Some(s), None) => Some(s),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        }
    }
}