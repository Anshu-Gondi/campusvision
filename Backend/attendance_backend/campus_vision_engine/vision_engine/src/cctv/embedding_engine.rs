use crate::app::AppState;
use crate::preprocessing::mat_to_array;
use ndarray::Array4;
use std::sync::Arc;
use opencv::prelude::*;

#[derive(Clone)]
pub struct EmbeddingEngine {
    state: Arc<AppState>,
    layout: String,
}

impl EmbeddingEngine {
    pub fn new(state: Arc<AppState>, layout: Option<String>) -> Self {
        Self {
            state,
            layout: layout.unwrap_or("NHWC".to_string()),
        }
    }

    pub async fn batch_embed(&self, mats: &[Mat]) -> anyhow::Result<Vec<Arc<Vec<f32>>>> {
        let mut batch: Vec<Array4<f32>> = Vec::with_capacity(mats.len());

        for mat in mats {
            use ndarray::Ix4;

            let arr = mat_to_array(mat, &self.layout)?.into_dimensionality::<Ix4>()?;

            batch.push(arr);
        }

        // 🔥 REAL batch inference
        let embeddings = self.state.face_pool.infer_batch(batch).await?;

        Ok(
            embeddings
                .into_iter()
                .map(|e| Arc::new(e))
                .collect()
        )
    }

    pub async fn batch_emotion(&self, mats: &[Mat]) -> anyhow::Result<Vec<Option<i64>>> {
        let mut batch = Vec::with_capacity(mats.len());

        for mat in mats {
            use ndarray::Ix4;

            let arr = mat_to_array(mat, &self.layout)?.into_dimensionality::<Ix4>()?;

            batch.push(arr);
        }

        let emotions = self.state.emotion_pool.infer_batch(batch).await?;

        Ok(
            emotions
                .into_iter()
                .map(|v|
                    v
                        .get(0)
                        .copied()
                        .map(|x| x as i64)
                )
                .collect()
        )
    }
}
