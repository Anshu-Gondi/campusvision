use crate::{
    models::{ort_model, tch_model},
    preprocess,
    hnsw_helper,
};
use opencv::{
    core::Size,
    imgcodecs,
};

#[derive(Debug)]
pub struct DetectionResult {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

pub fn detect_and_embed_rust(image_bytes: &[u8]) -> anyhow::Result<DetectionResult> {
    // Decode image
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::from_slice(image_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;

    // Detect faces
    let faces = preprocess::detect_faces(
        &mat,
        "models/face_detection_yunet_2023mar.onnx",
        Size::new(320, 320),
        0.6,
    )?;

    let best = preprocess::pick_best_face(&faces)?;

    if let Some((rect, landmarks)) = best {
        // Preprocess tensor
        let tensor =
            preprocess::preprocess_from_mat_and_landmarks(&mat, rect, &landmarks)?;

        // Run embedding model
        let embedding = match tch_model::run_face_model(&tensor) {
            Ok(v) => v,
            Err(_) => ort_model::run_face_model_onnx(&tensor)?,
        };

        Ok(DetectionResult {
            found: true,
            bbox: Some((rect.x, rect.y, rect.width, rect.height)),
            embedding: Some(embedding),
        })
    } else {
        Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        })
    }
}

pub fn detect_and_add_person_rust(
    image_bytes: &[u8],
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
) -> anyhow::Result<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let result = detect_and_embed_rust(image_bytes)?;

    if !result.found {
        anyhow::bail!("No face detected");
    }

    let embedding = result.embedding.expect("embedding missing");

    let id = hnsw_helper::add_face_embedding(
        embedding,
        name,
        person_id,
        roll_no,
        role,
    )?;

    Ok(id)
}
