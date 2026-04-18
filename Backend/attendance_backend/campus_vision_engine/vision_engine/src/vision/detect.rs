use crate::preprocessing;
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
    pub tier: Option<adaptive::QualityTier>, // 🔥 NEW
    pub warning: Option<String>, // 🔥 NEW
}

static LAST_EMBEDDING: Lazy<Mutex<Option<Vec<f32>>>> = Lazy::new(|| Mutex::new(None));

/* ============================================================
   INTERNAL ASYNC IMPLEMENTATION (POOL-BASED)
   ============================================================ */

async fn detect_and_embed_internal(
    state: Arc<AppState>,
    image_bytes: Vec<u8>,
    layout: Option<String>,
    enrollment: bool
) -> Result<DetectionResult> {
    // 1️⃣ Get model input size
    let (model_h, model_w) = state.face_pool.get_model_input_size()?; // if you exposed it
    // OR hardcode if fixed: (160, 160)

    // 2️⃣ Preprocess image
    let (face_mat, rect, _landmarks, face_roi) = match
        preprocessing::preprocess_image_dynamic(
            &image_bytes,
            Some((model_h, model_w)),
            state.yunet_pool.clone()
        ).await
    {
        Ok(res) => res,
        Err(_) => {
            return Ok(DetectionResult {
                found: false,
                bbox: None,
                embedding: None,
                tier: Some(adaptive::QualityTier::Reject),
                warning: Some("preprocess_failed".to_string()),
            });
        }
    };

    // 3️⃣ QUALITY CHECK (REAL PIPELINE)
    let quality = match quality::compute_quality(&face_roi, &rect) {
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

    // 4️⃣ DECISION ENGINE (CRITICAL)
    let decision = decision::evaluate(&quality, enrollment);

    // 🔥 NEW: adaptive layer (final authority)
    let adaptive = adaptive::adapt(&decision, enrollment);

    if !adaptive.proceed {
        return Ok(DetectionResult {
            found: false,
            bbox: None,
            embedding: None,
            tier: Some(adaptive.tier),
            warning: adaptive.warning,
        });
    }

    // 5️⃣ Convert Mat → ndarray
    // ✅ ArcFace-specific preprocessing — correct normalization + NHWC shape
    let input_array = preprocessing::mat_to_array_arcface(&face_mat)?;
    let input_array4: Array4<f32> = input_array
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| anyhow!("Expected 4D tensor"))?;

    // 🔥 6️⃣ INFERENCE THROUGH POOL (NON-BLOCKING)
    let embedding = state.face_pool.infer(input_array4).await?;

    // 7️⃣ Replay protection
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
        tier: Some(adaptive.tier), // 🔥 NEW
        warning: adaptive.warning, // 🔥 NEW
    })
}

/* ============================================================
   PUBLIC API
   ============================================================ */

pub async fn detect_and_embed_rust(
    state: Arc<AppState>,
    image_bytes: Vec<u8>,
    layout: Option<String>,
    enrollment: bool
) -> Result<DetectionResult> {
    detect_and_embed_internal(state, image_bytes, layout, enrollment).await
}

pub async fn detect_and_enroll_person_rust(
    state: Arc<AppState>,
    school_id: String,
    image_bytes: Vec<u8>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
    layout: Option<String>
) -> Result<usize> {
    // ─────────────────────────────
    // 1️⃣ VALIDATION
    // ─────────────────────────────
    if !["student", "teacher"].contains(&role.as_str()) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    // ─────────────────────────────
    // 2️⃣ DETECT + EMBED
    // ─────────────────────────────
    let result = detect_and_embed_rust(
        state.clone(),
        image_bytes,
        layout,
        true // enrollment mode
    ).await?;

    let embedding = result.embedding.ok_or_else(|| anyhow!("No valid face detected"))?;

    // ─────────────────────────────
    // 3️⃣ DUPLICATE CHECK (CRITICAL)
    // ─────────────────────────────
    let dup = crate::face_db::check_duplicate_rust(
        &school_id,
        &embedding,
        &role,
        0.75 // ⚠️ tune this
    );

    if dup.duplicate {
        return Err(
            anyhow!(
                "Duplicate face detected: {:?} (similarity {:.3})",
                dup.name,
                dup.similarity.unwrap_or(0.0)
            )
        );
    }

    // ─────────────────────────────
    // 4️⃣ ADD TO EMBEDDINGS (MULTI-TENANT)
    // ─────────────────────────────
    let id = embeddings::add_face_embedding(
        &school_id, // ✅ NEW (CRITICAL)
        embedding,
        name,
        person_id,
        roll_no,
        role
    )?;

    // ─────────────────────────────
    // 5️⃣ OPTIONAL: TRIGGER BACKUP (ASYNC)
    // ─────────────────────────────
    let state_clone = state.clone();
    let school_id_clone = school_id.clone();

    tokio::spawn(async move {
        let _ = state_clone.face_db_backup.save(&school_id_clone, "face_db").await;
    });

    Ok(id)
}
