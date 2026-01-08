use crate::{
    models::{ort_model, tch_model},
    preprocess::preprocess_image,
    utils::cosine_similarity,
};

/// Verify face by comparing embedding similarity
pub fn verify_face_rust(
    input_image: Vec<u8>,
    known_embedding: Vec<f32>,
) -> Result<f32, String> {
    let img_tensor =
        preprocess_image(&input_image).map_err(|e| e.to_string())?;

    let emb_vec = match tch_model::run_face_model(&img_tensor) {
        Ok(v) => v,
        Err(_) => ort_model::run_face_model_onnx(&img_tensor)
            .map_err(|e| e.to_string())?,
    };

    Ok(cosine_similarity(&emb_vec, &known_embedding))
}

/// Detect emotion from face image
pub fn detect_emotion_rust(
    input_image: Vec<u8>,
) -> Result<i64, String> {
    let img_tensor =
        preprocess_image(&input_image).map_err(|e| e.to_string())?;

    let result = match tch_model::run_emotion_model(&img_tensor) {
        Ok(r) => r,
        Err(_) => ort_model::run_emotion_model_onnx(&img_tensor)
            .map_err(|e| e.to_string())?,
    };

    Ok(result)
}
