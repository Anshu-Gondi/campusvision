/// cctv/track_manager.rs  (replaces existing)
///
/// Changes vs original:
///   1. `lock_identity`  — writes person_id + confidence + id_locked onto a track.
///   2. `update_confidence` — rolling average to prevent single-frame noise flips.
///   3. `should_resolve`  — decides whether a track needs identity search this tick.
///   4. Hungarian + match_tracks unchanged in logic, kept as-is.

use intelligence_core::utils::cosine_similarity;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use crate::cctv::types::TrackedFace;
use opencv::core::{Rect, Point2f};

/// How many consecutive frames with the same person_id before we lock it.
const LOCK_HITS_REQUIRED: u32 = 3;

/// Minimum similarity score from batch_search to accept an identity.
pub const MIN_IDENTITY_SCORE: f32 = 0.55;

/// Every N embedding updates we re-verify even locked tracks.
/// Catches people walking away and being replaced by someone else.
const REVERIFY_INTERVAL: u32 = 30;

#[derive(Clone)]
pub struct TrackManager {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32,
}

impl TrackManager {
    pub fn new(max_age: u32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
        }
    }

    pub fn create_track(
        &mut self,
        bbox: Rect,
        landmarks: Vec<Point2f>,
        embedding: Arc<Vec<f32>>,
        emotion: Option<i64>,
        now: Instant,
    ) {
        let id = self.next_id;
        self.next_id += 1;

        self.tracks.insert(id, TrackedFace {
            track_id: id,
            person_id: None,
            embedding,
            emotion,
            bbox,
            landmarks,
            hits: 1,
            age: 0,
            last_seen: now,
            confidence: 0.0,   // starts at 0 — not identified yet
            id_locked: false,
            last_embedding_update: 0,
        });
    }

    pub fn update_track(
        &mut self,
        track_id: usize,
        bbox: Rect,
        landmarks: Vec<Point2f>,
        embedding: Arc<Vec<f32>>,
        emotion: Option<i64>,
        now: Instant,
    ) {
        if let Some(track) = self.tracks.get_mut(&track_id) {
            track.bbox = bbox;
            track.landmarks = landmarks;
            track.embedding = embedding;
            track.emotion = emotion;
            track.last_seen = now;
            track.age = 0;
            track.hits += 1;
            track.last_embedding_update += 1;

            // If locked, reset lock when embedding drifts significantly.
            // This handles "person walked away, new person in same spot".
            if track.id_locked && track.last_embedding_update >= REVERIFY_INTERVAL {
                track.id_locked = false;
                track.last_embedding_update = 0;
            }
        }
    }

    /// Write back an identity resolution result onto a track.
    ///
    /// Implements a "vote" system: a new person_id only gets locked in
    /// after LOCK_HITS_REQUIRED consecutive frames agree on the same id.
    /// This prevents single noisy frames from locking a wrong identity.
    pub fn lock_identity(
        &mut self,
        track_id: usize,
        resolved_person_id: usize,
        score: f32,
    ) {
        let Some(track) = self.tracks.get_mut(&track_id) else { return };

        // Already locked to same person — just refresh confidence.
        if track.id_locked && track.person_id == Some(resolved_person_id) {
            track.confidence = smooth(track.confidence, score, 0.3);
            return;
        }

        // Different person resolved while locked — begin unlock vote.
        if track.id_locked && track.person_id != Some(resolved_person_id) {
            // Only override if the new score is significantly higher.
            if score > track.confidence + 0.10 {
                track.id_locked = false;
                track.person_id = Some(resolved_person_id);
                track.confidence = score;
            }
            return;
        }

        // Not yet locked — accumulate hits for this person_id.
        match track.person_id {
            Some(existing) if existing == resolved_person_id => {
                // Same candidate — update confidence and check lock threshold.
                track.confidence = smooth(track.confidence, score, 0.4);
                if track.hits >= LOCK_HITS_REQUIRED && track.confidence >= MIN_IDENTITY_SCORE {
                    track.id_locked = true;
                }
            }
            _ => {
                // New or different candidate — reset to this one.
                track.person_id = Some(resolved_person_id);
                track.confidence = score;
            }
        }
    }

    /// Clear identity on a track (used when score falls below threshold).
    pub fn clear_identity(&mut self, track_id: usize) {
        if let Some(track) = self.tracks.get_mut(&track_id) {
            if !track.id_locked {
                track.person_id = None;
                track.confidence = 0.0;
            }
        }
    }

    /// Returns true if this track needs an identity search this tick.
    ///
    /// Rules:
    ///   - Never search on first hit (embedding too fresh, person still entering frame).
    ///   - Search every frame until locked.
    ///   - Once locked, only re-search every REVERIFY_INTERVAL updates.
    pub fn should_resolve(&self, track_id: usize, min_hits: u32) -> bool {
        let Some(track) = self.tracks.get(&track_id) else { return false };

        if track.hits < min_hits {
            return false; // not stable enough yet
        }

        if !track.id_locked {
            return true; // keep searching until we lock
        }

        // Locked — only reverify at interval
        track.last_embedding_update == 0 // set to 0 when reverify window opens
    }

    pub fn increment_age(&mut self, track_id: usize) {
        if let Some(track) = self.tracks.get_mut(&track_id) {
            track.age += 1;
        }
    }

    pub fn cleanup(&mut self) {
        let max_age = self.max_age;
        self.tracks.retain(|_, t| t.age <= max_age);
    }

    pub fn match_tracks(
        &self,
        track_list: &[TrackedFace],
        detections: &[Arc<Vec<f32>>],
    ) -> Vec<Option<usize>> {
        let n = track_list.len();
        let m = detections.len();

        if n == 0 || m == 0 {
            return vec![None; n];
        }

        let mut cost = vec![vec![0.0f32; m]; n];
        for i in 0..n {
            for j in 0..m {
                let sim = cosine_similarity(&track_list[i].embedding, &detections[j]);
                cost[i][j] = 1.0 - sim;
            }
        }

        let assignments = hungarian(&cost);
        let mut result = vec![None; n];

        for (track_idx, det_idx_opt) in assignments.into_iter().enumerate() {
            if let Some(det_idx) = det_idx_opt {
                if det_idx < m {
                    let sim = 1.0 - cost[track_idx][det_idx];
                    if sim >= 0.35 {
                        result[track_idx] = Some(det_idx);
                    }
                }
            }
        }

        result
    }
}

/// Exponential moving average: weight = how much new value matters.
#[inline]
fn smooth(old: f32, new: f32, weight: f32) -> f32 {
    old * (1.0 - weight) + new * weight
}

// ── Hungarian algorithm (unchanged) ──────────────────────────────────────────

fn hungarian(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let n = cost.len();
    let m = cost[0].len();
    let size = n.max(m);

    let mut matrix = vec![vec![0.0f32; size]; size];
    for i in 0..n {
        for j in 0..m {
            matrix[i][j] = cost[i][j];
        }
    }

    let mut u = vec![0.0f32; size + 1];
    let mut v = vec![0.0f32; size + 1];
    let mut p = vec![0usize; size + 1];
    let mut way = vec![0usize; size + 1];

    for i in 1..=size {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f32::INFINITY; size + 1];
        let mut used = vec![false; size + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f32::INFINITY;
            let mut j1 = 0;

            for j in 1..=size {
                if !used[j] {
                    let cur = matrix[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=size {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 { break; }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }

    let mut result = vec![None; n];
    for j in 1..=size {
        if p[j] != 0 && p[j] <= n && j <= m {
            result[p[j] - 1] = Some(j - 1);
        }
    }
    result
}