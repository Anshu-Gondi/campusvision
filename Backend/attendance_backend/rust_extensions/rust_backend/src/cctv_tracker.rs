// src/cctv_tracker.rs
use opencv::core::{Point2f, Rect};
use std::collections::HashMap;
use std::time::Instant;
use crate::utils::cosine_similarity;

#[derive(Clone)]
pub struct TrackedFace {
    pub track_id: usize,
    pub person_id: Option<usize>,        // None = unknown, Some = identified
    pub embedding: Vec<f32>,
    pub bbox: Rect,
    pub landmarks: Vec<Point2f>,
    pub hits: u32,
    pub age: u32,
    pub last_seen: Instant,
}

pub struct FaceTracker {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32,        // frames before removal
}

impl FaceTracker {
    pub fn new(max_age: u32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
        }
    }

    pub fn update(&mut self, detections: Vec<(Rect, Vec<Point2f>, Vec<f32>)>) -> Vec<TrackedFace> {
        let now = Instant::now();

        let mut matched = vec![false; detections.len()];
        let mut active: Vec<TrackedFace> = self.tracks.values().cloned().collect();

        // Match existing tracks
        for track in active.iter_mut() {
            let mut best_idx = None;
            let mut best_score = 0.3f32; // IoU + similarity weight

            for (i, (bbox, _, emb)) in detections.iter().enumerate() {
                if matched[i] { continue; }

                let iou_val = iou(&track.bbox, bbox);
                let sim = cosine_similarity(&track.embedding, emb);
                let score = iou_val * 0.5 + sim * 0.5;

                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }

            if let Some(i) = best_idx {
                let (bbox, landmarks, emb) = &detections[i];
                track.bbox = *bbox;
                track.landmarks = landmarks.clone();
                track.embedding = emb.clone();
                track.hits += 1;
                track.age = 0;
                track.last_seen = now;
                matched[i] = true;
            } else {
                track.age += 1;
            }
        }

        // Create new tracks for unmatched detections
        for (i, (bbox, landmarks, emb)) in detections.into_iter().enumerate() {
            if !matched[i] {
                active.push(TrackedFace {
                    track_id: self.next_id,
                    person_id: None,
                    embedding: emb,
                    bbox,
                    landmarks,
                    hits: 1,
                    age: 0,
                    last_seen: now,
                });
                self.next_id += 1;
            }
        }

        // Remove dead tracks
        active.retain(|t| t.age <= self.max_age);

        // Update internal map
        self.tracks = active.iter().enumerate().map(|(i, t)| (i, t.clone())).collect();
        active
    }
}

fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    if x2 <= x1 || y2 <= y1 { return 0.0; }
    let inter = (x2 - x1) * (y2 - y1);
    let union = a.area() + b.area() - inter;
    inter as f32 / union as f32
}