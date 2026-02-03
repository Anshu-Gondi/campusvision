// src/cctv/tracker.rs

use intelligence_core::embeddings::batch_search;
use crate::models::onnx_models::{
    run_emotion_model_onnx as ort_emotion,
    run_face_model_onnx as ort_face,
};
use crate::preprocessing::preprocess_image;
use intelligence_core::utils::cosine_similarity;
use opencv::core::{ Point2f, Rect };
use opencv::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use ndarray::{ Array4, ArrayBase, Dim, OwnedRepr };

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
    pub confidence: f32, // 0.0 → 1.0
    pub id_locked: bool, // once true, ID never changes
}

pub struct FaceTracker {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32, // frames before removal
}

/// Convert OpenCV Mat to Array4<f32> (batch=1, channels=3, H, W)
fn mat_to_array4(mat: &Mat) -> anyhow::Result<Array4<f32>> {
    if mat.channels() != 3 {
        anyhow::bail!("Expected 3-channel BGR image, got {}", mat.channels());
    }

    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;

    let mut array = Array4::<f32>::zeros((1, 3, rows, cols));

    let data = mat.data_bytes()?;

    for y in 0..rows {
        for x in 0..cols {
            let idx = (y * cols + x) * 3;
            array[[0, 0, y, x]] = data[idx + 2] as f32 / 255.0; // R
            array[[0, 1, y, x]] = data[idx + 1] as f32 / 255.0; // G
            array[[0, 2, y, x]] = data[idx + 0] as f32 / 255.0; // B
        }
    }

    Ok(array)
}

/// Generate face embedding from Mat
#[inline]
fn run_face_embedding(mat: &opencv::prelude::Mat) -> Vec<f32> {
    match mat_to_array4(mat) {
        Ok(array) => ort_face(&array).unwrap_or_default(),
        Err(_) => vec![],
    }
}

/// Run emotion model from Mat
#[inline]
fn run_emotion(mat: &opencv::prelude::Mat) -> i64 {
    match mat_to_array4(mat) {
        Ok(array) => ort_emotion(&array).unwrap_or(-1),
        Err(_) => -1,
    }
}

impl FaceTracker {
    pub fn new(max_age: u32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
        }
    }

    /// Main update loop: accepts raw image bytes (BGR) per frame
    pub fn update_from_bytes(&mut self, frames: Vec<&[u8]>) -> Vec<TrackedFace> {
        let now = Instant::now();
        let mut detections: Vec<(Rect, Vec<Point2f>, opencv::prelude::Mat)> = Vec::new();

        // Step 0: preprocess each frame → Mat + bbox + landmarks
        for frame_bytes in frames {
            if let Ok((mat, bbox, landmarks)) = preprocess_image(frame_bytes) {
                detections.push((bbox, landmarks, mat));
            }
        }

        // Step 1: Embedding + emotion extraction
        let embeddings_emotions: Vec<(Vec<f32>, i64, Rect, Vec<Point2f>)> = detections
            .iter()
            .map(|(_, _, mat)| {
                let emb = run_face_embedding(mat);
                let emo = run_emotion(mat);
                (emb, emo, detections[0].0, detections[0].1.clone()) // FIXME: use correct bbox/landmarks
            })
            .collect();

        let mut matched = vec![false; embeddings_emotions.len()];
        let mut active: Vec<TrackedFace> = self.tracks.values().cloned().collect();

        // Step 1.5: Batch HNSW search (using intelligence_core)
        let all_embeddings: Vec<Vec<f32>> = embeddings_emotions
            .iter()
            .map(|(emb, _, _, _)| emb.clone())
            .collect();
        let student_hits = batch_search(&all_embeddings, "student", 5);
        let teacher_hits = batch_search(&all_embeddings, "teacher", 3);

        // Step 2: Match existing tracks (same as before)
        for track in active.iter_mut() {
            let mut best_idx = None;
            let mut best_score = 0.3f32;

            for (i, (emb, _, _, _)) in embeddings_emotions.iter().enumerate() {
                if matched[i] {
                    continue;
                }

                let sim = cosine_similarity(&track.embedding, emb);
                if sim > best_score {
                    best_score = sim;
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

                // HNSW + confidence logic (same as before)
                if !track.id_locked && track.hits >= 3 {
                    let s = student_hits[i];
                    let t = teacher_hits[i];
                    if let Some((id, sim)) = s.or(t) {
                        if sim >= 0.75 {
                            track.person_id = Some(id);
                            track.confidence = sim;
                        }
                    }
                }

                if track.person_id.is_some() {
                    track.confidence = (track.confidence + 0.1).min(1.0);
                    if track.confidence >= 0.85 && track.hits >= 5 {
                        track.id_locked = true;
                    }
                }
            } else {
                track.age += 1;
                track.confidence *= 0.95;
            }
        }

        // Step 3: Add new tracks from unmatched
        for (i, (emb, emo, bbox, landmarks)) in embeddings_emotions.into_iter().enumerate() {
            if !matched[i] {
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
                    confidence: if person_id.is_some() {
                        0.6
                    } else {
                        0.0
                    },
                    id_locked: false,
                });
                self.next_id += 1;
            }
        }

        // Step 4: Remove old tracks
        active.retain(|t| t.age <= self.max_age);

        // Step 5: Update internal map
        self.tracks = active
            .iter()
            .map(|t| (t.track_id, t.clone()))
            .collect();
        active
    }
}
