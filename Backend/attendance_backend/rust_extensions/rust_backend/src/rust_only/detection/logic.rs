use crate::{
    hnsw_helper,
    models::{ort_model, tch_model},
    preprocess,
};
use opencv::{core::Size, imgcodecs};

#[derive(Debug)]
pub struct DetectionResult {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

pub fn detect_and_embed_rust(
    image_bytes: &[u8],
    model_path: Option<&str>, // optional
) -> anyhow::Result<DetectionResult> {
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::from_slice(image_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;

    let model_path = match model_path {
        Some(p) if !p.is_empty() => Some(p),
        _ => None,
    };

    let faces = preprocess::detect_faces(&mat, model_path, Size::new(320, 320), 0.6)?;
    let best = preprocess::pick_best_face(&faces)?;

    if let Some((rect, landmarks)) = best {
        if landmarks.len() < 2 {
            return Ok(DetectionResult {
                found: false,
                bbox: None,
                embedding: None,
            });
        }

        let tensor = preprocess::preprocess_from_mat_and_landmarks(&mat, rect, &landmarks)?;
        let embedding = tch_model::run_face_model(&tensor).or_else(|e1| {
            ort_model::run_face_model_onnx(&tensor)
                .map_err(|e2| anyhow::anyhow!("TorchScript error: {}; ONNX error: {}", e1, e2))
        })?;

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

    let result = detect_and_embed_rust(image_bytes, None)?;

    if !result.found {
        anyhow::bail!("No face detected");
    }

    let embedding = result.embedding.expect("embedding missing");

    let id = hnsw_helper::add_face_embedding(embedding, name, person_id, roll_no, role)?;

    Ok(id)
}
