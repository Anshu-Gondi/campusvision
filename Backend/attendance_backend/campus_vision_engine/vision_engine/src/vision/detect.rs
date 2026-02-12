use crate::preprocessing;
use intelligence_core::embeddings;
use crate::models::onnx_models as ort_model;

use opencv::prelude::*;
use anyhow::{anyhow, Result};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use intelligence_core::utils::cosine_similarity;

use tokio::task;

use ndarray::Array4;

#[derive(Debug)]
pub struct DetectionResult {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

static LAST_EMBEDDING: Lazy<Mutex<Option<Vec<f32>>>> =
    Lazy::new(|| Mutex::new(None));

/* ============================================================
   INTERNAL BLOCKING IMPLEMENTATION
   ============================================================ */

fn detect_and_embed_blocking(
    image_bytes: &[u8],
    model_input_size: Option<(usize, usize)>,
    layout: Option<String>,
    enrollment: bool,
) -> Result<DetectionResult> {

    // 1️⃣ Preprocess
    let (face_mat, rect, _landmarks) =
        match preprocessing::preprocess_image_dynamic(
            image_bytes,
            model_input_size,
        ) {
            Ok(res) => res,
            Err(_) => {
                return Ok(DetectionResult {
                    found: false,
                    bbox: None,
                    embedding: None,
                });
            }
        };

    // 2️⃣ Face area sanity check
    let img_area = face_mat.rows() * face_mat.cols();
    let face_area = face_mat.rows() * face_mat.cols();

    if face_area < img_area / 20 {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        });
    }

    // 3️⃣ Blur check
    {
        use opencv::{
            core::{mean_std_dev, AlgorithmHint},
            imgproc,
        };

        let mut gray = opencv::core::Mat::default();
        imgproc::cvt_color(
            &face_mat,
            &mut gray,
            imgproc::COLOR_RGB2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut lap = opencv::core::Mat::default();
        imgproc::laplacian(
            &gray,
            &mut lap,
            opencv::core::CV_64F,
            1,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        let mut mean = opencv::core::Scalar::default();
        let mut stddev = opencv::core::Scalar::default();

        mean_std_dev(&lap, &mut mean, &mut stddev, &opencv::core::no_array())?;

        if stddev[0] * stddev[0] < 30.0 {
            return Ok(DetectionResult {
                found: false,
                bbox: None,
                embedding: None,
            });
        }
    }

    // 4️⃣ Convert to ndarray
    let input_layout = layout.unwrap_or("NHWC".to_string());
    let input_array =
        preprocessing::mat_to_array(&face_mat, &input_layout)?;

    let input_array4: Array4<f32> = input_array
    .into_dimensionality()
    .map_err(|_| anyhow!("Expected 4D tensor"))?;

    let embedding =
        ort_model::run_face_model_onnx(
            &input_array4,
            "models/facenet.onnx",
        )?;

    // 5️⃣ Replay protection
    if !enrollment {
        let mut last = LAST_EMBEDDING.lock().unwrap();

        if let Some(prev) = &*last {
            if cosine_similarity(prev, &embedding) >= 0.995 {
                return Ok(DetectionResult {
                    found: false,
                    bbox: None,
                    embedding: None,
                });
            }
        }

        *last = Some(embedding.clone());
    }

    Ok(DetectionResult {
        found: true,
        bbox: Some((rect.x, rect.y, rect.width, rect.height)),
        embedding: Some(embedding),
    })
}

/* ============================================================
   PUBLIC ASYNC API (AXUM SAFE)
   ============================================================ */

pub async fn detect_and_embed_rust(
    image_bytes: Vec<u8>,
    model_input_size: Option<(usize, usize)>,
    layout: Option<String>,
    enrollment: bool,
) -> Result<DetectionResult> {

    task::spawn_blocking(move || {
        detect_and_embed_blocking(
            &image_bytes,
            model_input_size,
            layout,
            enrollment,
        )
    })
    .await
    .map_err(|e| anyhow!("Join error: {e}"))?
}

pub async fn detect_and_add_person_rust(
    image_bytes: Vec<u8>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
    model_input_size: Option<(usize, usize)>,
    layout: Option<String>,
) -> Result<usize> {

    if !["student", "teacher"].contains(&role.as_str()) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let result = detect_and_embed_rust(
        image_bytes,
        model_input_size,
        layout,
        true,
    ).await?;

    let embedding =
        result.embedding.ok_or_else(|| anyhow!("Embedding missing"))?;

    let id = embeddings::add_face_embedding(
        embedding,
        name,
        person_id,
        roll_no,
        role,
    )?;

    Ok(id)
}