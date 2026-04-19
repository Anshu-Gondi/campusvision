use crate::app::AppState;
use crate::preprocessing::{mat_to_arcface_input, mat_to_emotion_input};
use ndarray::Array4;
use std::sync::Arc;
use opencv::prelude::*;
use anyhow::Result;

#[derive(Clone)]
pub struct EmbeddingEngine {
    state: Arc<AppState>,
}

impl EmbeddingEngine {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// Batch embed faces using ArcFace (correct normalization + NHWC)
    pub async fn batch_embed(&self, mats: &[Mat]) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut batch: Vec<Array4<f32>> = Vec::with_capacity(mats.len());

        for mat in mats {
            // Use the new correct ArcFace preprocessing
            let tensor = mat_to_arcface_input(mat)?;
            batch.push(tensor);
        }

        // Real batch inference through the pool
        let embeddings = self.state.face_pool.infer_batch(batch).await?;

        // Wrap each embedding in Arc for efficient sharing
        Ok(
            embeddings
                .into_iter()
                .map(|e| Arc::new(e))
                .collect()
        )
    }

    /// Batch emotion detection using Emotion FERPlus
    pub async fn batch_emotion(&self, mats: &[Mat]) -> Result<Vec<Option<i64>>> {
        let mut batch: Vec<Array4<f32>> = Vec::with_capacity(mats.len());

        for mat in mats {
            // Use the new correct Emotion preprocessing (grayscale NCHW)
            let tensor = mat_to_emotion_input(mat)?;
            batch.push(tensor);
        }

        // Run batch inference
        let emotions = self.state.emotion_pool.infer_batch(batch).await?;

        // Convert raw output to emotion class index (0-7)
        Ok(
            emotions
                .into_iter()
                .map(|scores| {
                    if scores.is_empty() {
                        None
                    } else {
                        // Find index with highest score (argmax)
                        scores
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(idx, _)| idx as i64)
                    }
                })
                .collect()
        )
    }

    /// Single face embedding (convenience method)
    pub async fn embed_single(&self, mat: &Mat) -> Result<Vec<f32>> {
        let tensor = mat_to_arcface_input(mat)?;
        self.state.face_pool.run_face_embedding(tensor).await
    }

    /// Single emotion detection (convenience method)
    pub async fn emotion_single(&self, mat: &Mat) -> Result<i64> {
        let tensor = mat_to_emotion_input(mat)?;
        self.state.emotion_pool.run_emotion(tensor).await
    }
}