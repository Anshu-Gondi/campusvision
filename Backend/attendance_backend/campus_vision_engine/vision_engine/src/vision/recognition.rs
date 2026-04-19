use crate::preprocessing::{preprocess_image_dynamic, mat_to_arcface_input, mat_to_emotion_input};
use crate::app::AppState;
use std::sync::Arc;
use intelligence_core::utils::cosine_similarity;
use anyhow::{ anyhow, Result };
use ndarray::Array4;
use opencv::prelude::*;

/// Verify face by comparing embedding similarity
pub async fn verify_face_with_pool(
    state: Arc<AppState>,
    input_image: Vec<u8>,
    known_embedding: Vec<f32>,
) -> Result<f32> {
    // 1. Full preprocessing: detect + align → returns RGB Mat (112x112)
    let (aligned_rgb, _, _, _) = preprocess_image_dynamic(
        &input_image,
        state.yunet_pool.clone(),
    ).await?;

    // 2. Convert to ArcFace input format (NHWC + special normalization)
    let input_tensor: Array4<f32> = mat_to_arcface_input(&aligned_rgb)?;

    // 3. Run inference
    let embedding = state.face_pool.run_face_embedding(input_tensor).await?;

    // 4. Compute similarity
    let similarity = cosine_similarity(&embedding, &known_embedding);

    Ok(similarity)
}

/// Detect emotion from face image
pub async fn detect_emotion_with_pool(
    state: Arc<AppState>,
    input_image: Vec<u8>,
) -> Result<i64> {
    // 1. Full preprocessing: detect + align
    let (_, _, _, face_roi) = preprocess_image_dynamic(
        &input_image,
        state.yunet_pool.clone(),
    ).await?;

    // 2. Convert to Emotion input format (grayscale NCHW 64x64)
    let input_tensor: Array4<f32> = mat_to_emotion_input(&face_roi)?;

    // 3. Run emotion inference
    let emotion_id = state.emotion_pool.run_emotion(input_tensor).await?;

    Ok(emotion_id)
}