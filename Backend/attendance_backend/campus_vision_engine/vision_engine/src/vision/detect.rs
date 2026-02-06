use crate::preprocessing;
use intelligence_core::embeddings;
use crate::models::onnx_models as ort_model;
use opencv::{ core::{ Size, Mat, mean_std_dev, AlgorithmHint }, imgcodecs, imgproc };
use opencv::prelude::*;
use anyhow::{ anyhow, Result };
use std::cell::RefCell;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use intelligence_core::utils::cosine_similarity;
use ndarray::Array4;

thread_local! {
    static MAT_BUF: RefCell<Mat> = RefCell::new(Mat::default());
}

static LAST_EMBEDDING: Lazy<Mutex<Option<Vec<f32>>>> = Lazy::new(|| Mutex::new(None));

#[derive(Debug)]
pub struct DetectionResult {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

/// Convert OpenCV Mat (BGR, u8) → NCHW Array4<f32>
fn mat_to_array4(mat: &Mat) -> Result<Array4<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;

    let data = mat.data_bytes()?;
    let mut arr = Array4::<f32>::zeros((1, 3, rows, cols));

    for y in 0..rows {
        for x in 0..cols {
            let idx = (y * cols + x) * 3;
            arr[[0, 0, y, x]] = (data[idx] as f32) / 255.0;
            arr[[0, 1, y, x]] = (data[idx + 1] as f32) / 255.0;
            arr[[0, 2, y, x]] = (data[idx + 2] as f32) / 255.0;
        }
    }

    Ok(arr)
}

/// Detect faces → pick best → align → ONNX embedding
pub fn detect_and_embed_rust(
    image_bytes: &[u8],
    model_path: Option<&str>
) -> Result<DetectionResult> {
    // ─────────────────────────────
    // 1. Decode image (FIXED)
    // ─────────────────────────────
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::from_slice(image_bytes),
        imgcodecs::IMREAD_COLOR
    )?;

    if mat.empty() {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        });
    }

    let yunet_path = model_path.filter(|p| !p.is_empty());

    // ─────────────────────────────
    // 2. Face detection
    // ─────────────────────────────
    let faces = preprocessing::detect_faces(&mat, yunet_path.as_deref(), Size::new(320, 320), 0.6)?;

    let Some((rect, landmarks)) = preprocessing::pick_best_face(&faces)? else {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        });
    };

    // ─────────────────────────────
    // Face size sanity check
    // ─────────────────────────────
    let img_area = mat.rows() * mat.cols();
    let face_area = rect.width * rect.height;
    if face_area < img_area / 7 {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        });
    }

    // ─────────────────────────────
    // Blur check
    // ─────────────────────────────
    let mut gray = Mat::default();
    imgproc::cvt_color(
        &mat,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;

    let mut lap = Mat::default();
    imgproc::laplacian(
        &gray,
        &mut lap,
        opencv::core::CV_64F,
        1,
        1.0,
        0.0,
        opencv::core::BORDER_DEFAULT
    )?;

    let mut mean = opencv::core::Scalar::default();
    let mut stddev = opencv::core::Scalar::default();
    mean_std_dev(&lap, &mut mean, &mut stddev, &opencv::core::no_array())?;

    if stddev[0] * stddev[0] < 80.0 {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
        });
    }

    // ─────────────────────────────
    // 4. Align face
    // ─────────────────────────────
    let face_mat = preprocessing::align_face(&mat, rect, &landmarks)?;

    // ─────────────────────────────
    // 5. ONNX embedding (FIXED)
    // ─────────────────────────────
    let input = mat_to_array4(&face_mat)?;
    let embedding = ort_model::run_face_model_onnx(&input)?;

    // ─────────────────────────────
    // Replay protection
    // ─────────────────────────────
    {
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

/// Detect → embed → add to index
pub fn detect_and_add_person_rust(
    image_bytes: &[u8],
    name: String,
    person_id: u64,
    roll_no: String,
    role: String
) -> Result<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let result = detect_and_embed_rust(image_bytes, None)?;
    let embedding = result.embedding.ok_or_else(|| anyhow!("Embedding missing"))?;

    let id = embeddings::add_face_embedding(embedding, name, person_id, roll_no, role)?;

    Ok(id)
}
