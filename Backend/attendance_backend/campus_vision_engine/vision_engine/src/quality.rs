use anyhow::{anyhow, Result};
use opencv::{
    core::{self, Mat, Scalar},
    imgproc,
    prelude::*,
};
use opencv::core::AlgorithmHint;

pub struct QualityMetrics {
    pub blur: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub face_area_ratio: f32,
}

pub fn compute_quality(mat: &Mat, face_rect: &opencv::core::Rect) -> Result<QualityMetrics> {
    if mat.empty() {
        return Err(anyhow!("empty_mat"));
    }

    // ==========================
    // 1️⃣ GRAYSCALE
    // ==========================
    let mut gray = Mat::default();
    imgproc::cvt_color(mat, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // ==========================
    // 2️⃣ BLUR
    // ==========================
    let mut lap = Mat::default();
    imgproc::laplacian(
        &gray,
        &mut lap,
        core::CV_64F,
        1,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(&lap, &mut mean, &mut stddev, &core::no_array())?;

    let blur = (stddev[0] * stddev[0]) as f32;

    if blur < 10.0 {
        return Err(anyhow!("extreme_blur"));
    }

    // ==========================
    // 3️⃣ BRIGHTNESS
    // ==========================
    let mean_scalar = core::mean(&gray, &core::no_array())?;
    let brightness = mean_scalar[0] as f32;

    // ==========================
    // 4️⃣ CONTRAST
    // ==========================
    let mut mean2 = Scalar::default();
    let mut stddev2 = Scalar::default();
    core::mean_std_dev(&gray, &mut mean2, &mut stddev2, &core::no_array())?;

    let contrast = stddev2[0] as f32;

    if contrast < 10.0 {
        return Err(anyhow!("low_contrast"));
    }

    // ==========================
    // 5️⃣ FACE SIZE RATIO
    // ==========================
    let area_face = (face_rect.width * face_rect.height) as f32;
    let area_img = (mat.cols() * mat.rows()) as f32;

    if area_img <= 0.0 {
        return Err(anyhow!("invalid_image_area"));
    }

    let ratio = area_face / area_img;

    // 🔴 NEW: reject absurd cases early
    if ratio < 0.02 {
        return Err(anyhow!("face_too_small_extreme"));
    }

    Ok(QualityMetrics {
        blur,
        brightness,
        contrast,
        face_area_ratio: ratio,
    })
}