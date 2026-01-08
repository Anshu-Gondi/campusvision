// src/cctv_tracker.rs

use crate::hnsw_helper::batch_search;
use crate::models::ort_model::{
    run_emotion_model_onnx as ort_emotion, run_face_model_onnx as ort_face,
};
use crate::models::tch_model::{run_emotion_model as tch_emotion, run_face_model as tch_face};
use crate::utils::{cosine_similarity, iou};
use opencv::core::{Point2f, Rect};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone)]
pub struct TrackedFace {
    pub track_id: usize,
    pub person_id: Option<usize>, // None = unknown, Some = identified
    pub embedding: Vec<f32>,
    pub emotion: Option<i64>,
    pub bbox: Rect,
    pub landmarks: Vec<Point2f>,
    pub hits: u32,
    pub age: u32,
    pub last_seen: Instant,
}

pub struct FaceTracker {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32, // frames before removal
}

#[inline]
fn run_face_embedding(tensor: &tch::Tensor) -> Vec<f32> {
    // Try TorchScript first (faster if warm)
    if let Ok(emb) = tch_face(tensor) {
        if !emb.is_empty() {
            return emb;
        }
    }

    // Fallback to ONNX
    ort_face(tensor).unwrap_or_default()
}

#[inline]
fn run_emotion(tensor: &tch::Tensor) -> i64 {
    // Try TorchScript first
    if let Ok(e) = tch_emotion(tensor) {
        if e >= 0 {
            return e;
        }
    }

    // Fallback to ONNX
    ort_emotion(tensor).unwrap_or(-1)
}

impl FaceTracker {
    pub fn new(max_age: u32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
        }
    }

    /// Main update loop: detections = Vec<(Rect, landmarks, tensor)>
    pub fn update(
        &mut self,
        detections: Vec<(Rect, Vec<Point2f>, tch::Tensor)>,
    ) -> Vec<TrackedFace> {
        let now = Instant::now();

        // Step 1: Parallel embedding + emotion extraction
        let embeddings_emotions: Vec<(Vec<f32>, i64, Rect, Vec<Point2f>)> = detections
            .iter()
            .map(|(bbox, landmarks, tensor)| {
                let emb = run_face_embedding(tensor);
                let emo = run_emotion(tensor);
                (emb, emo, *bbox, landmarks.clone())
            })
            .collect();

        let mut matched = vec![false; embeddings_emotions.len()];
        let mut active: Vec<TrackedFace> = self.tracks.values().cloned().collect();

        // STEP 1.5: Batch HNSW search (ONE LOCK PER ROLE)
        let all_embeddings: Vec<Vec<f32>> = embeddings_emotions
            .iter()
            .map(|(emb, _, _, _)| emb.clone())
            .collect();

        let student_hits = batch_search(&all_embeddings, "student", 5);
        let teacher_hits = batch_search(&all_embeddings, "teacher", 3);

        // Step 2: Match existing tracks
        for track in active.iter_mut() {
            let mut best_idx = None;
            let mut best_score = 0.3f32; // IoU + similarity weight

            for (i, (emb, _, bbox, _)) in embeddings_emotions.iter().enumerate() {
                if matched[i] {
                    continue;
                }

                let iou_val = iou(&track.bbox, bbox);
                let sim = cosine_similarity(&track.embedding, emb);
                let score = iou_val * 0.5 + sim * 0.5;

                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }

            if let Some(i) = best_idx {
                let (emb, emo, bbox, landmarks) = &embeddings_emotions[i];
                track.bbox = *bbox;
                track.landmarks = landmarks.clone();
                track.embedding = emb.clone();
                track.emotion = Some(*emo);
                track.hits += 1;
                track.age = 0;
                track.last_seen = now;
                matched[i] = true;

                // Optional: HNSW search in parallel
                if track.person_id.is_none() {
                    let s = student_hits[i];
                    let t = teacher_hits[i];
                    track.person_id = s.or(t).map(|(id, _)| id);
                }
            } else {
                track.age += 1;
            }
        }

        // Step 3: Add unmatched detections as new tracks
        for (i, (emb, emo, bbox, landmarks)) in embeddings_emotions.into_iter().enumerate() {
            if !matched[i] {
                // optional: HNSW search for new track
                let s = student_hits[i];
                let t = teacher_hits[i];
                let person_id = s.or(t).map(|(id, _)| id);

                active.push(TrackedFace {
                    track_id: self.next_id,
                    person_id,
                    embedding: emb,
                    emotion: Some(emo),
                    bbox,
                    landmarks,
                    hits: 1,
                    age: 0,
                    last_seen: now,
                });
                self.next_id += 1;
            }
        }

        // Step 4: Remove old tracks
        active.retain(|t| t.age <= self.max_age);

        // Step 5: Update internal map
        self.tracks = active
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.clone()))
            .collect();
        active
    }
}
