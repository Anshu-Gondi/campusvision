use crate::{models::onnx_models, preprocessing::preprocess_image_dynamic};
use intelligence_core::utils::cosine_similarity;
use anyhow::{anyhow, Result};
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
        _ => return Err(anyhow!("Unsupported layout: {}", layout)),
    };

    for y in 0..rows {
        for x in 0..cols {
            let idx = (y * cols + x) * 3;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;

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
pub fn verify_face_onnx(
    input_image: Vec<u8>,
    known_embedding: &[f32],
    model_input_size: Option<(usize, usize)>,
    layout: Option<&str>,
) -> Result<f32> {
    let layout_str = layout.unwrap_or("NHWC");

    // preprocess → aligned RGB face
    let (face_mat, _, _) = preprocess_image_dynamic(&input_image, model_input_size)?;

    // convert OpenCV Mat → ndarray
    let input_tensor = mat_to_array4_dynamic(&face_mat, layout_str)?;

    // run FaceNet ONNX
    let emb_vec = onnx_models::run_face_model_onnx(&input_tensor, "models/facenet.onnx")?;

    Ok(cosine_similarity(&emb_vec, known_embedding))
}

/// Detect emotion from face image
pub fn detect_emotion_onnx(
    input_image: Vec<u8>,
    model_input_size: Option<(usize, usize)>,
    layout: Option<&str>,
) -> Result<i64> {
    let layout_str = layout.unwrap_or("NHWC");

    let (face_mat, _, _) = preprocess_image_dynamic(&input_image, model_input_size)?;
    let input_tensor = mat_to_array4_dynamic(&face_mat, layout_str)?;

    onnx_models::run_emotion_model_onnx(&input_tensor, "models/emotion.onnx")
}
