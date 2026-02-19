use std::sync::Arc;
use std::time::Instant;
use crate::app::AppState;

use crate::cctv::detector::Detector;
use crate::cctv::embedding_engine::EmbeddingEngine;
use crate::cctv::identity_resolver::IdentityResolver;
use crate::cctv::track_manager::TrackManager;
use crate::cctv::types::TrackedFace;

#[derive(Clone)]
pub struct FaceTracker {
    detector: Detector,
    embedder: EmbeddingEngine,
    resolver: IdentityResolver,
    pub manager: TrackManager,
}

impl FaceTracker {
    pub fn new(
        max_age: u32,
        state: Arc<AppState>,
        model_input_size: Option<(usize, usize)>,
        layout: Option<String>
    ) -> Self {
        Self {
            detector: Detector::new(state.clone()),
            embedder: EmbeddingEngine::new(state, layout),
            resolver: IdentityResolver,
            manager: TrackManager::new(max_age),
        }
    }

    pub async fn update_from_bytes(
        &mut self,
        frames: Vec<&[u8]>
    ) -> anyhow::Result<Vec<TrackedFace>> {
        let now = Instant::now();

        // 1️⃣ Detection
        let detections = self.detector.detect(frames).await?;
        if detections.is_empty() {
            return Ok(vec![]);
        }

        let mats: Vec<_> = detections
            .iter()
            .map(|(_, _, m)| m.clone())
            .collect();

        // 2️⃣ Batch embedding
        let embeddings = self.embedder.batch_embed(&mats).await?;
        let emotions = self.embedder.batch_emotion(&mats).await?;
        // 3️⃣ Hungarian match
        let active: Vec<_> = self.manager.tracks.values().cloned().collect();

        let assignment = self.manager.match_tracks(&active, &embeddings);

        // Track which detections were matched
        let mut matched_detections = vec![false; embeddings.len()];

        // 4️⃣ Apply assignments
        for (track_idx, maybe_det_idx) in assignment.iter().enumerate() {
            if let Some(det_idx) = maybe_det_idx {
                matched_detections[*det_idx] = true;

                let (bbox, landmarks, _) = &detections[*det_idx];
                let embedding = &embeddings[*det_idx];
                let emotion = emotions.get(*det_idx).cloned().unwrap_or(None);

                self.manager.update_track(
                    active[track_idx].track_id,
                    bbox.clone(),
                    landmarks.clone(),
                    embedding.clone(),
                    emotion,
                    now
                );
            } else {
                // No match → age track
                self.manager.increment_age(active[track_idx].track_id);
            }
        }

        // 5 Create new tracks for unmatched detections
        for (i, matched) in matched_detections.iter().enumerate() {
            if !matched {
                let (bbox, landmarks, _) = &detections[i];

                self.manager.create_track(
                    bbox.clone(),
                    landmarks.clone(),
                    embeddings[i].clone(),
                    emotions.get(i).cloned().unwrap_or(None),
                    now
                );
            }
        }

        // 6 Cleanup dead tracks
        self.manager.cleanup();

        Ok(self.manager.tracks.values().cloned().collect())
    }
}
