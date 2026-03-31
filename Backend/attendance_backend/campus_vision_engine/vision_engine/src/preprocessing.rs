use anyhow::{ anyhow, Result };
use opencv::{
    core::{ self, Mat, Point2f, Rect, Scalar, Size, Vector, AlgorithmHint },
    imgcodecs,
    imgproc,
    calib3d::{ estimate_affine_partial_2d, RANSAC },
    prelude::*,
};
use std::sync::Arc;

use crate::models::yunet_pool::YuNetPool;

pub const DEFAULT_YUNET_MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";

/// Preprocess image bytes → returns (aligned_face_mat, bbox, landmarks)
pub async fn preprocess_image_dynamic(
    image_bytes: &[u8],
    model_input_size: Option<(usize, usize)>,
    yunet_pool: Arc<YuNetPool>
) -> Result<(Mat, Rect, Vec<Point2f>)> {
    // 1️⃣ Decode image
    let mut mat = imgcodecs::imdecode(&Vector::from_slice(image_bytes), imgcodecs::IMREAD_COLOR)?;

    if mat.empty() {
        return Err(anyhow!("Decoded image is empty"));
    }

    // 2️⃣ Only downscale EXTREME images (4K+)
    let (w, h) = (mat.cols(), mat.rows());

    if w > 1280 || h > 1280 {
        let scale = 1280.0 / (w.max(h) as f64);

        let new_size = Size::new(((w as f64) * scale) as i32, ((h as f64) * scale) as i32);

        let mut resized = Mat::default();
        imgproc::resize(&mat, &mut resized, new_size, 0.0, 0.0, imgproc::INTER_AREA)?;
        mat = resized;
    }

    // 3️⃣ Detect faces via pool
    let faces = yunet_pool.detect(&mat).await?;

    let (face_rect, landmarks) = pick_best_face(&faces)?.ok_or_else(||
        anyhow!("No face detected")
    )?;

    // 4️⃣ Align face
    let aligned = align_face(
        &mat,
        face_rect,
        &landmarks,
        model_input_size.map(|(h, w)| (w as i32, h as i32))
    )?;

    Ok((aligned, face_rect, landmarks))
}

/// Select highest score face
pub fn pick_best_face(faces: &Mat) -> Result<Option<(Rect, Vec<Point2f>)>> {
    if faces.empty() || faces.rows() == 0 {
        return Ok(None);
    }

    let mut best_idx = -1;

    let mut best_score = -1.0;

    for row in 0..faces.rows() {
        let score = *faces.at_2d::<f32>(row, 14)?;
        let w = *faces.at_2d::<f32>(row, 2)?;
        let h = *faces.at_2d::<f32>(row, 3)?;

        let combined = score * (w * h);

        if combined > best_score {
            best_score = combined;
            best_idx = row;
        }
    }

    if best_score < 0.7 {
        return Ok(None);
    }
    if best_idx < 0 {
        return Ok(None);
    }

    let x = *faces.at_2d::<f32>(best_idx, 0)? as i32;
    let y = *faces.at_2d::<f32>(best_idx, 1)? as i32;
    let w = *faces.at_2d::<f32>(best_idx, 2)? as i32;
    let h = *faces.at_2d::<f32>(best_idx, 3)? as i32;

    let bbox = Rect::new(x, y, w, h);

    if w < 40 || h < 40 {
        return Ok(None);
    }

    let mut landmarks = Vec::with_capacity(5);
    for &col in &[4, 6, 8, 10, 12] {
        let px = *faces.at_2d::<f32>(best_idx, col)?;
        let py = *faces.at_2d::<f32>(best_idx, col + 1)?;
        landmarks.push(Point2f::new(px, py));
    }

    Ok(Some((bbox, landmarks)))
}

/// Align face using 2-eye affine transform
pub fn align_face(
    mat: &Mat,
    mut face_rect: Rect,
    landmarks: &[Point2f],
    target_size: Option<(i32, i32)>
) -> Result<Mat> {
    if landmarks.len() < 2 {
        return Err(anyhow!("Not enough landmarks"));
    }

    let img_w = mat.cols();
    let img_h = mat.rows();

    face_rect.x = face_rect.x.clamp(0, img_w - 1);
    face_rect.y = face_rect.y.clamp(0, img_h - 1);
    face_rect.width = face_rect.width.min(img_w - face_rect.x);
    face_rect.height = face_rect.height.min(img_h - face_rect.y);

    if face_rect.width <= 0 || face_rect.height <= 0 {
        return Err(anyhow!("Invalid face rect"));
    }

    let pad = 0.2; // 20% padding

    let cx = face_rect.x + face_rect.width / 2;
    let cy = face_rect.y + face_rect.height / 2;

    let new_w = ((face_rect.width as f32) * (1.0 + pad)) as i32;
    let new_h = ((face_rect.height as f32) * (1.0 + pad)) as i32;

    let mut x = cx - new_w / 2;
    let mut y = cy - new_h / 2;

    x = x.clamp(0, mat.cols() - 1);
    y = y.clamp(0, mat.rows() - 1);

    let w = new_w.min(mat.cols() - x);
    let h = new_h.min(mat.rows() - y);

    let padded_rect = Rect::new(x, y, w, h);
    let face_roi = Mat::roi(mat, padded_rect)?;

    let (out_w, out_h) = target_size.unwrap_or((112, 112));

    // Canonical eye placement
    let desired_left_eye = Point2f::new((out_w as f32) * 0.35, (out_h as f32) * 0.4);
    let desired_right_eye = Point2f::new((out_w as f32) * 0.65, (out_h as f32) * 0.4);

    let left_eye = Point2f::new(
        landmarks[1].x - (padded_rect.x as f32),
        landmarks[1].y - (padded_rect.y as f32)
    );

    let right_eye = Point2f::new(
        landmarks[0].x - (padded_rect.x as f32),
        landmarks[0].y - (padded_rect.y as f32)
    );

    let src = Vector::from_slice(&[left_eye, right_eye]);
    let dst = Vector::from_slice(&[desired_left_eye, desired_right_eye]);

    let warp_mat = estimate_affine_partial_2d(
        &src,
        &dst,
        &mut core::no_array(),
        RANSAC,
        3.0,
        2000,
        0.99,
        10
    )?;

    if warp_mat.empty() {
        let mut resized = Mat::default();

        imgproc::resize(
            &face_roi,
            &mut resized,
            Size::new(out_w, out_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR
        )?;

        return Ok(resized);
    }

    let mut aligned = Mat::default();

    imgproc::warp_affine(
        &face_roi,
        &mut aligned,
        &warp_mat,
        Size::new(out_w, out_h),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0)
    )?;

    // Convert BGR → RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        &aligned,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;

    Ok(rgb)
}

/// Convert Mat → ndarray (NHWC or NCHW)
pub fn mat_to_array(mat: &Mat, layout: &str) -> Result<ndarray::ArrayD<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let data = mat.data_bytes()?;

    match layout {
        "NHWC" => {
            let arr = ndarray::Array::from_shape_fn((1, rows, cols, 3), |(_, y, x, c)| {
                (data[(y * cols + x) * 3 + c] as f32) / 255.0
            });
            Ok(arr.into_dyn())
        }
        "NCHW" => {
            let arr = ndarray::Array::from_shape_fn((1, 3, rows, cols), |(_, c, y, x)| {
                (data[(y * cols + x) * 3 + c] as f32) / 255.0
            });
            Ok(arr.into_dyn())
        }
        _ => Err(anyhow!("Unsupported layout")),
    }
}
