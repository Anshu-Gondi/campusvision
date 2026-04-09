use opencv::{core::Mat, imgproc, prelude::*};

pub struct QualityMetrics {
    pub blur: f32,
    pub brightness: f32,
    pub face_area_ratio: f32,
}

pub fn compute_quality(mat: &Mat, face_rect: &opencv::core::Rect) -> anyhow::Result<QualityMetrics> {
    // Blur (variance of Laplacian)
    let mut gray = Mat::default();
    imgproc::cvt_color(mat, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut lap = Mat::default();
    imgproc::laplacian(&gray, &mut lap, opencv::core::CV_64F, 1, 1.0, 0.0, opencv::core::BORDER_DEFAULT)?;

    let mut mean = opencv::core::Scalar::default();
    let mut stddev = opencv::core::Scalar::default();
    opencv::core::mean_std_dev(&lap, &mut mean, &mut stddev, &opencv::core::no_array())?;

    let blur = (stddev[0] * stddev[0]) as f32;

    // Brightness
    let mean_scalar = opencv::core::mean(&gray, &opencv::core::no_array())?;
    let brightness = mean_scalar[0] as f32;

    // Face size ratio
    let area_face = (face_rect.width * face_rect.height) as f32;
    let area_img = (mat.cols() * mat.rows()) as f32;

    let ratio = area_face / area_img;

    Ok(QualityMetrics {
        blur,
        brightness,
        face_area_ratio: ratio,
    })
}