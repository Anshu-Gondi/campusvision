use crate::preprocessing::{ preprocess_image_dynamic, mat_to_arcface_input };
use crate::quality;
use crate::decision;
use crate::adaptive;
use crate::app::AppState;

use intelligence_core::embeddings;
use intelligence_core::utils::cosine_similarity;

use opencv::prelude::*;
use anyhow::{ anyhow, Result };
use std::sync::{ Arc, Mutex };
use once_cell::sync::Lazy;
use ndarray::Array4;

#[derive(Debug)]
pub struct DetectionResult {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
    pub tier: Option<adaptive::QualityTier>,
    pub warning: Option<String>,
}

static LAST_EMBEDDING: Lazy<Mutex<Option<Vec<f32>>>> = Lazy::new(|| Mutex::new(None));

/* ============================================================
   INTERNAL ASYNC IMPLEMENTATION
   ============================================================ */

async fn detect_and_embed_internal(
    state: Arc<AppState>,
    image_bytes: Vec<u8>,
    enrollment: bool
) -> Result<DetectionResult> {
    // 1️⃣ Preprocess: Detection + Alignment (new powerful function)
    let (aligned_rgb, rect, _landmarks, _face_roi) = match
        preprocess_image_dynamic(&image_bytes, state.yunet_pool.clone()).await
    {
        Ok(res) => res,
        Err(e) => {
            return Ok(DetectionResult {
                found: false,
                bbox: None,
                embedding: None,
                tier: Some(adaptive::QualityTier::Reject),
                warning: Some(format!("preprocess_failed: {}", e)),
            });
        }
    };

    // 2️⃣ Quality Assessment
    let quality = match quality::compute_quality(&_face_roi, &rect) {
        // Note: you may need to adjust this call
        Ok(q) => q,
        Err(e) => {
            return Ok(DetectionResult {
                found: false,
                bbox: None,
                embedding: None,
                tier: Some(adaptive::QualityTier::Reject),
                warning: Some(e.to_string()),
            });
        }
    };

    // 3️⃣ Decision + Adaptive Layer
    let decision = decision::evaluate(&quality, enrollment);
    let adaptive_result = adaptive::adapt(&decision, enrollment);

    if !adaptive_result.proceed {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
            tier: Some(adaptive_result.tier),
            warning: adaptive_result.warning,
        });
    }

    // 4️⃣ Convert aligned RGB Mat → ArcFace tensor (correct normalization)
    let input_tensor: Array4<f32> = mat_to_arcface_input(&aligned_rgb)?;

    // 5️⃣ Run inference through pool
    let embedding = state.face_pool.infer(input_tensor).await?;

    // 6️⃣ Replay / Duplicate frame protection
    if !enrollment {
        let mut last = LAST_EMBEDDING.lock().unwrap();
        if let Some(prev) = &*last {
            if cosine_similarity(prev, &embedding) >= 0.995 {
                return Ok(DetectionResult {
                    found: false,
                    bbox: None,
                    embedding: None,
                    tier: Some(adaptive::QualityTier::Low),
                    warning: Some("duplicate_frame".to_string()),
                });
            }
        }
        *last = Some(embedding.clone());
    }

    Ok(DetectionResult {
        found: true,
        bbox: Some((rect.x, rect.y, rect.width, rect.height)),
        embedding: Some(embedding),
        tier: Some(adaptive_result.tier),
        warning: adaptive_result.warning,
    })
}

/* ============================================================
   PUBLIC API
   ============================================================ */

/// Main public function for detection + embedding
pub async fn detect_and_embed_rust(
    state: Arc<AppState>,
    image_bytes: Vec<u8>,
    _layout: Option<String>, // kept for backward compatibility if needed
    enrollment: bool
) -> Result<DetectionResult> {
    detect_and_embed_internal(state, image_bytes, enrollment).await
}

/// Enrollment flow
pub async fn detect_and_enroll_person_rust(
    state: Arc<AppState>,
    school_id: String,
    image_bytes: Vec<u8>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
    _layout: Option<String> // kept for compatibility
) -> Result<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let result = detect_and_embed_rust(state.clone(), image_bytes, None, true).await?;

    let embedding = result.embedding.ok_or_else(|| anyhow!("No valid face detected"))?;

    // Duplicate check
    let dup = crate::face_db::check_duplicate_rust(&school_id, &embedding, &role, 0.75);

    if dup.duplicate {
        return Err(
            anyhow!(
                "Duplicate face detected: {:?} (similarity {:.3})",
                dup.name,
                dup.similarity.unwrap_or(0.0)
            )
        );
    }

    // Add to embeddings database
    let id = embeddings::add_face_embedding(&school_id, embedding, name, person_id, roll_no, role)?;

    // Async backup
    let state_clone = state.clone();
    let school_id_clone = school_id.clone();
    tokio::spawn(async move {
        let _ = state_clone.face_db_backup.save(&school_id_clone, "face_db").await;
    });

    Ok(id)
}
