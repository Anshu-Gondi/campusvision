use intelligence_core::embeddings::batch_search;
use crate::models::onnx_models::{
    run_emotion_model_onnx as ort_emotion,
    run_face_model_onnx as ort_face,
};
use crate::preprocessing;
use intelligence_core::utils::cosine_similarity;
use opencv::core::{ Point2f, Rect };
use opencv::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use ndarray::Array4;

const EMBED_EVERY_N_FRAMES: u32 = 5;
const EMOTION_EVERY_N_FRAMES: u32 = 10;

#[derive(Clone)]
pub struct TrackedFace {
    pub track_id: usize,
    pub person_id: Option<usize>,
    pub embedding: Vec<f32>,
    pub emotion: Option<i64>,
    pub bbox: Rect,
    pub landmarks: Vec<Point2f>,
    pub hits: u32,
    pub age: u32,
    pub last_seen: Instant,
    pub confidence: f32,
    pub id_locked: bool,
    pub last_embedding_update: u32,
}

pub struct FaceTracker {
    pub tracks: HashMap<usize, TrackedFace>,
    pub next_id: usize,
    pub max_age: u32,
    pub face_model_path: String,
    pub emotion_model_path: String,
    pub model_input_size: Option<(usize, usize)>,
    pub layout: Option<String>,
}

fn mat_to_array4(mat: &Mat, layout: &str) -> anyhow::Result<Array4<f32>> {
    if mat.channels() != 3 {
        anyhow::bail!("Expected 3-channel BGR image");
    }

    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let mut array = match layout {
        "NCHW" => Array4::<f32>::zeros((1, 3, rows, cols)),
        "NHWC" => Array4::<f32>::zeros((1, rows, cols, 3)),
        _ => anyhow::bail!("Unsupported layout: {}", layout),
    };

    let data = mat.data_bytes()?;

    for y in 0..rows {
        for x in 0..cols {
            let i = (y * cols + x) * 3;
            let r = (data[i + 2] as f32) / 255.0;
            let g = (data[i + 1] as f32) / 255.0;
            let b = (data[i + 0] as f32) / 255.0;

            match layout {
                "NCHW" => {
                    array[[0, 0, y, x]] = r;
                    array[[0, 1, y, x]] = g;
                    array[[0, 2, y, x]] = b;
                }
                "NHWC" => {
                    array[[0, y, x, 0]] = r;
                    array[[0, y, x, 1]] = g;
                    array[[0, y, x, 2]] = b;
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(array)
}

#[inline]
fn run_face_embedding(mat: &Mat, model_path: &str, layout: &str) -> Vec<f32> {
    let input = mat_to_array4(mat, layout).expect("Failed to convert Mat to Array4");

    ort_face(&input, model_path).expect("Face ONNX inference failed")
}

#[inline]
fn run_emotion(mat: &Mat, model_path: &str, layout: &str) -> i64 {
    mat_to_array4(mat, layout)
        .ok()
        .and_then(|a| ort_emotion(&a, model_path).ok())
        .unwrap_or(-1)
}

impl FaceTracker {
    pub fn new(
        max_age: u32,
        face_model_path: String,
        emotion_model_path: String,
        model_input_size: Option<(usize, usize)>,
        layout: Option<String>
    ) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
            face_model_path,
            emotion_model_path,
            model_input_size,
            layout,
        }
    }

    pub fn update_from_bytes(&mut self, frames: Vec<&[u8]>) -> Vec<TrackedFace> {
        let now = Instant::now();
        let layout_str = self.layout.as_deref().unwrap_or("NHWC");

        // Step 0: detection
        let mut detections: Vec<(Rect, Vec<Point2f>, Mat)> = Vec::new();
        for bytes in frames {
            if
                let Ok((mat, bbox, landmarks)) = preprocessing::preprocess_image_dynamic(
                    bytes,
                    self.model_input_size
                )
            {
                detections.push((bbox, landmarks, mat));
            }
        }

        let mut active: Vec<TrackedFace> = self.tracks.values().cloned().collect();
        let mut matched = vec![false; detections.len()];

        // Step 1: match existing tracks
        for track in active.iter_mut() {
            let mut best_idx = None;
            let mut best_score = 0.35;

            for (i, (_, _, mat)) in detections.iter().enumerate() {
                if matched[i] {
                    continue;
                }

                let should_embed =
                    track.embedding.is_empty() ||
                    track.hits.saturating_sub(track.last_embedding_update) >= EMBED_EVERY_N_FRAMES;

                let emb = if should_embed {
                    run_face_embedding(mat, &self.face_model_path, layout_str)
                } else {
                    track.embedding.as_slice().to_vec()
                };

                if track.embedding.is_empty() || emb.is_empty() {
                    continue;
                }

                let sim = cosine_similarity(&track.embedding, &emb);
                if sim > best_score {
                    best_score = sim;
                    best_idx = Some((i, emb, should_embed));
                }
            }

            if let Some((i, emb, emb_updated)) = best_idx {
                let (bbox, landmarks, mat) = &detections[i];
                track.bbox = *bbox;
                track.landmarks = landmarks.clone();
                track.hits += 1;
                track.age = 0;
                track.last_seen = now;
                matched[i] = true;

                if emb_updated {
                    track.embedding = emb;
                    track.last_embedding_update = track.hits;
                }

                let run_emo = !track.id_locked && track.hits % EMOTION_EVERY_N_FRAMES == 0;
                if run_emo {
                    track.emotion = Some(run_emotion(mat, &self.emotion_model_path, layout_str));
                }

                if !track.id_locked && emb_updated && track.hits >= 3 {
                    let student_hits = batch_search(&[track.embedding.clone()], "student", 1);
                    let teacher_hits = batch_search(&[track.embedding.clone()], "teacher", 1);

                    let best_hit = match (student_hits.get(0), teacher_hits.get(0)) {
                        (Some(Some(s)), Some(Some(t))) => {
                            if s.1 >= t.1 { Some(s) } else { Some(t) }
                        }
                        (Some(Some(s)), _) => Some(s),
                        (_, Some(Some(t))) => Some(t),
                        _ => None,
                    };

                    if let Some((id, sim)) = best_hit {
                        if *sim >= 0.75 {
                            track.person_id = Some(*id);
                            track.confidence = *sim;
                        }
                    }
                }

                if track.person_id.is_some() {
                    track.confidence = (track.confidence + 0.08).min(1.0);
                    if track.confidence >= 0.85 && track.hits >= 5 {
                        track.id_locked = true;
                    }
                }
            } else {
                track.age += 1;
                track.confidence *= 0.95;
            }
        }

        // Step 2: add new tracks
        for (i, (bbox, landmarks, mat)) in detections.into_iter().enumerate() {
            if !matched[i] {
                let emb = run_face_embedding(&mat, &self.face_model_path, layout_str);
                let emo = run_emotion(&mat, &self.emotion_model_path, layout_str);

                if emb.is_empty() {
                    continue;
                }

                active.push(TrackedFace {
                    track_id: self.next_id,
                    person_id: None,
                    embedding: emb,
                    emotion: Some(emo),
                    bbox,
                    landmarks,
                    hits: 1,
                    age: 0,
                    last_seen: now,
                    confidence: 0.0,
                    id_locked: false,
                    last_embedding_update: 1,
                });
                self.next_id += 1;
            }
        }

        // Step 3: cleanup
        active.retain(|t| t.age <= self.max_age);
        self.tracks = active
            .iter()
            .map(|t| (t.track_id, t.clone()))
            .collect();

        active
    }
}
