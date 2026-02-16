use lapjv::lapjv;
use intelligence_core::utils::cosine_similarity;

pub struct TrackManager {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32,
}

impl TrackManager {
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

        let mut cost = vec![vec![1.0; m]; n];

        for (i, t) in track_list.iter().enumerate() {
            for (j, emb) in detections.iter().enumerate() {
                let sim = cosine_similarity(&t.embedding, emb);
                cost[i][j] = 1.0 - sim;
            }
        }

        let raw_assignment = lapjv(&cost).unwrap();

        let mut result = vec![None; n];

        for (track_idx, &det_idx) in raw_assignment.iter().enumerate() {
            if det_idx < m {
                let sim = 1.0 - cost[track_idx][det_idx];

                // Similarity gate
                if sim >= 0.35 {
                    result[track_idx] = Some(det_idx);
                }
            }
        }

        result
    }
}
