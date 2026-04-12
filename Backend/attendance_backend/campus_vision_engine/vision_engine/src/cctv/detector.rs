use crate::app::AppState;
use opencv::core::{ Rect, Point2f, Mat };
use std::sync::Arc;

#[derive(Clone)]
pub struct Detector {
    state: Arc<AppState>,
}

impl Detector {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    pub async fn detect(
        &self,
        frames: Vec<&[u8]>
    ) -> anyhow::Result<Vec<(Rect, Vec<Point2f>, Mat)>> {
        let mut results = Vec::new();

        for bytes in frames {
            if
                let Ok((mat, bbox, landmarks, _)) = crate::preprocessing::preprocess_image_dynamic(
                    bytes,
                    None,
                    self.state.yunet_pool.clone()
                ).await
            {
                results.push((bbox, landmarks, mat));
            }
        }

        Ok(results)
    }
}
