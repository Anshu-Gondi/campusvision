use intelligence_core::utils::cosine_similarity;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use crate::cctv::types::TrackedFace;
use opencv::core::{ Rect, Point2f };

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
        now: Instant
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
            confidence: 1.0,
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
        now: Instant
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
        }
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
        detections: &[Arc<Vec<f32>>]
    ) -> Vec<Option<usize>> {
        let n = track_list.len();
        let m = detections.len();

        if n == 0 || m == 0 {
            return vec![None; n];
        }

        // Build cost matrix
        let mut cost = vec![vec![0.0f32; m]; n];

        for i in 0..n {
            for j in 0..m {
                let sim = cosine_similarity(&track_list[i].embedding, &detections[j]);
                cost[i][j] = 1.0 - sim; // convert similarity to cost
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

fn hungarian(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let n = cost.len();
    let m = cost[0].len();
    let size = n.max(m);

    // Create square matrix
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

            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
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

