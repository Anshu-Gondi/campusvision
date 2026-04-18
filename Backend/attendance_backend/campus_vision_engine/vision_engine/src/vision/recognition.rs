use crate::preprocessing::preprocess_image_dynamic;
use crate::app::AppState;
use std::sync::Arc;
use intelligence_core::utils::cosine_similarity;
use anyhow::{ anyhow, Result };
use ndarray::Array4;
use opencv::prelude::*;

/// Convert OpenCV Mat (RGB, u8) → NCHW or NHWC Array4<f32>
pub fn mat_to_array4_dynamic(mat: &opencv::core::Mat, layout: &str) -> Result<Array4<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;

    if mat.typ() != opencv::core::CV_8UC3 {
        return Err(anyhow!("Expected CV_8UC3 Mat"));
    }

    let data = mat.data_bytes()?;

    let mut array = match layout {
        "NCHW" => Array4::<f32>::zeros((1, 3, rows, cols)),
        "NHWC" => Array4::<f32>::zeros((1, rows, cols, 3)),
        _ => {
            return Err(anyhow!("Unsupported layout: {}", layout));
        }
    };

    for y in 0..rows {
        for x in 0..cols {
            let idx = (y * cols + x) * 3;
            let r = (data[idx] as f32) / 255.0;
            let g = (data[idx + 1] as f32) / 255.0;
            let b = (data[idx + 2] as f32) / 255.0;

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

/// Verify face by comparing embedding similarity
/// Verify face by comparing embedding similarity
pub async fn verify_face_with_pool(
    state: Arc<AppState>,
    input_image: Vec<u8>,
    known_embedding: Vec<f32>,
    model_input_size: Option<(usize, usize)>,
    layout: Option<&str>
) -> Result<f32> {
    let layout_str = layout.unwrap_or("NHWC");

    // 1️⃣ Preprocess (FIXED)
    let (face_mat, _, _, _) = preprocess_image_dynamic(
        &input_image,
        model_input_size,
        state.yunet_pool.clone()
    ).await?;

    // 2️⃣ Convert → tensor
    let input_tensor = crate::preprocessing::mat_to_array_arcface(&face_mat)?
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| anyhow!("tensor shape error"))?;

    // 3️⃣ Send to inference pool
    let embedding = state.face_pool.run_face_embedding(input_tensor).await?;

    // 4️⃣ Compute similarity (pure CPU)
    Ok(cosine_similarity(&embedding, &known_embedding))
}

/// Detect emotion from face image
pub async fn detect_emotion_with_pool(
    state: Arc<AppState>,
    input_image: Vec<u8>,
    model_input_size: Option<(usize, usize)>,
    layout: Option<&str>
) -> Result<i64> {
    let layout_str = layout.unwrap_or("NHWC");

    // 1️⃣ Preprocess (FIXED)
    let (face_mat, _, _, _) = preprocess_image_dynamic(
        &input_image,
        model_input_size,
        state.yunet_pool.clone()
    ).await?;

    // 2️⃣ Convert → tensor
    let input_tensor = mat_to_array4_dynamic(&face_mat, layout_str)?;

    // 3️⃣ Send to inference pool
    state.emotion_pool.run_emotion(input_tensor).await
}