/// cctv/detector.rs  (replaces existing)
///
/// Fix: original looped frames sequentially — so Vec<&[u8]> gave zero
/// parallelism benefit. Now all frames are dispatched concurrently via
/// `futures::future::join_all`, each going to the YuNet pool independently.
///
/// This matters most when a single tracker update call is given multiple
/// frames (e.g. from a high-fps camera sending burst updates).
/// Even for single-frame calls there is no overhead vs the old version.

use crate::app::AppState;
use opencv::core::{Mat, Point2f, Rect};
use std::sync::Arc;

#[derive(Clone)]
pub struct Detector {
    state: Arc<AppState>,
}

impl Detector {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// Detect faces in all frames concurrently.
    /// Returns one result per frame that had a valid face.
    /// Frames with no face or failed preprocessing are silently skipped.
    pub async fn detect(
        &self,
        frames: Vec<&[u8]>,
    ) -> anyhow::Result<Vec<(Rect, Vec<Point2f>, Mat)>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        // Dispatch all frames to the YuNet pool concurrently.
        // Each call to preprocess_image_dynamic is independent — they only
        // share the Arc<YuNetPool> which is already designed for concurrent use.
        let futs: Vec<_> = frames
            .into_iter()
            .map(|bytes| {
                let state = Arc::clone(&self.state);
                let bytes = bytes.to_vec(); // owned copy for the async task
                async move {
                    crate::preprocessing::preprocess_image_dynamic(
                        &bytes,
                        None,
                        state.yunet_pool.clone(),
                    )
                    .await
                    .ok() // failed detections become None, not errors
                    .map(|(mat, bbox, landmarks, _face_roi)| (bbox, landmarks, mat))
                }
            })
            .collect();

        // Run all detections concurrently on the Tokio executor.
        // YuNet pool workers are on dedicated OS threads so this is true
        // parallelism, not just interleaving.
        let results = futures::future::join_all(futs).await;

        Ok(results.into_iter().flatten().collect())
    }
}