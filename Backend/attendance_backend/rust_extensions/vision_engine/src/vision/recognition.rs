use crate::{ models::onnx_models, preprocessing::preprocess_image };
use intelligence_core::utils::cosine_similarity;

use anyhow::{anyhow, Result};
use ndarray::Array4;
use opencv::prelude::*;

/// Convert OpenCV Mat (RGB, u8) → NCHW Array4<f32>
pub fn mat_to_ndarray(mat: &opencv::core::Mat) -> Result<Array4<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;

    // ❌ NO `?` here — typ() returns i32
    if mat.typ() != opencv::core::CV_8UC3 {
        return Err(anyhow!("Expected CV_8UC3 Mat"));
    }

    let data = mat.data_bytes()?; // this DOES return Result

    let mut array = Array4::<f32>::zeros((1, 3, rows, cols));

    for y in 0..rows {
        for x in 0..cols {
            let idx: usize = (y * cols + x) * 3;

            array[[0, 0, y, x]] = data[idx] as f32 / 255.0;
            array[[0, 1, y, x]] = data[idx + 1] as f32 / 255.0;
            array[[0, 2, y, x]] = data[idx + 2] as f32 / 255.0;
        }
    }

    Ok(array)
}

/// Verify face by comparing embedding similarity
pub fn verify_face_onnx(input_image: Vec<u8>, known_embedding: &[f32]) -> Result<f32> {
    // preprocess → aligned RGB face
    let (face_mat, _, _) = preprocess_image(&input_image)?;

    // convert OpenCV Mat → ndarray
    let input_tensor = mat_to_ndarray(&face_mat)?;

    // run FaceNet ONNX
    let emb_vec = onnx_models::run_face_model_onnx(&input_tensor)?;

    Ok(cosine_similarity(&emb_vec, known_embedding))
}

/// Detect emotion from face image
pub fn detect_emotion_onnx(input_image: Vec<u8>) -> Result<i64> {
    let (face_mat, _, _) = preprocess_image(&input_image)?;

    let input_tensor = mat_to_ndarray(&face_mat)?;

    onnx_models::run_emotion_model_onnx(&input_tensor)
}
