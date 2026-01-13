use anyhow::{anyhow, Result};
use opencv::core::Ptr;
use opencv::{
    core::{self, AlgorithmHint, Mat, Point2f, Rect, Scalar, Size, Vector},
    imgcodecs, imgproc,
    objdetect::FaceDetectorYN,
    prelude::*,
};
use std::cell::RefCell;
use tch::Tensor;

thread_local! {
    static YUNET_CACHE: RefCell<Option<Ptr<FaceDetectorYN>>> = RefCell::new(None);
}

/// Target size for FaceNet / InsightFace embedding models
const FACE_NET_SIZE: i32 = 160;

/// Default YuNet model path (runtime file, NOT embedded)
pub const DEFAULT_YUNET_MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";

/// Preprocess image bytes → normalized [1, 3, 160, 160] tensor ready for FaceNet
/// This version DETECTS the face (calls YuNet), then aligns and returns the tensor.
pub fn preprocess_image(image_bytes: &[u8]) -> Result<Tensor> {
    // 1. Decode image (BGR)
    let mat = imgcodecs::imdecode(&Vector::from_slice(image_bytes), imgcodecs::IMREAD_COLOR)
        .map_err(|_| anyhow!("Failed to decode image bytes"))?;
    if mat.empty() {
        return Err(anyhow!("Empty image after decoding"));
    }

    // 2. Detect faces with YuNet
    let input_size = Size::new(320, 320); // speed/accuracy trade-off
    let faces = detect_faces(&mat, None, input_size, 0.6)?;
    let best = pick_best_face(&faces)?.ok_or_else(|| anyhow!("No face detected"))?;
    let (face_rect, landmarks) = best;

    preprocess_from_mat_and_landmarks(&mat, face_rect, &landmarks)
}

/// Preprocess given an already-detected face rectangle + landmarks on the original Mat.
/// This is used to avoid double detection when you already detected faces (detect_and_embed).
pub fn preprocess_from_mat_and_landmarks(
    mat: &Mat,
    mut face_rect: Rect,
    landmarks: &[Point2f],
) -> Result<Tensor> {
    // 1. Clamp bounding box to image bounds
    let img_w = mat.cols();
    let img_h = mat.rows();
    face_rect.x = face_rect.x.max(0);
    face_rect.y = face_rect.y.max(0);
    face_rect.width = (face_rect.x + face_rect.width).min(img_w) - face_rect.x;
    face_rect.height = (face_rect.y + face_rect.height).min(img_h) - face_rect.y;

    if face_rect.width <= 0 || face_rect.height <= 0 {
        return Err(anyhow!("Invalid face bounding box after clamping"));
    }

    // 2. Extract face ROI
    let face_roi = Mat::roi(mat, face_rect).map_err(|_| anyhow!("Failed to extract face ROI"))?;

    // 3. Eye positions relative to the ROI
    let eye_right = Point2f::new(
        landmarks[0].x - face_rect.x as f32,
        landmarks[0].y - face_rect.y as f32,
    );
    let eye_left = Point2f::new(
        landmarks[1].x - face_rect.x as f32,
        landmarks[1].y - face_rect.y as f32,
    );

    // 4. Desired eye positions in the final 160×160 image
    let desired_right_eye = Point2f::new(48.0, 48.0);
    let desired_left_eye = Point2f::new(112.0, 48.0);

    // 5. Compute affine transform to align eyes
    let src_points = vec![eye_right, eye_left];
    let dst_points = vec![desired_right_eye, desired_left_eye];
    let warp_mat = imgproc::get_affine_transform(
        &Vector::from_slice(&src_points),
        &Vector::from_slice(&dst_points),
    )?;

    // 6. Warp + resize to 160×160
    let mut aligned = Mat::default();
    imgproc::warp_affine(
        &face_roi,
        &mut aligned,
        &warp_mat,
        Size::new(FACE_NET_SIZE, FACE_NET_SIZE),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    // 7. Convert BGR → RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        &aligned,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 8. Convert to tch::Tensor and normalize for FaceNet
    // Ensure data exist
    let data_u8 = rgb.data_typed::<u8>()?;
    let tensor = Tensor::from_data_size(
        data_u8,
        &[FACE_NET_SIZE as i64, FACE_NET_SIZE as i64, 3],
        tch::Kind::Uint8,
    );

    let tensor_f32 = tensor.to_kind(tch::Kind::Float) / 255.0;
    let normalized = (tensor_f32 - 0.5) / 0.5; // (x - 0.5)/0.5 == (x*2 -1)

    // Final shape: [1, 3, 160, 160]
    Ok(normalized.unsqueeze(0).permute(&[0, 3, 1, 2]))
}

// ---------------------------------------------------------------------------
// YuNet face detection helpers (unchanged except comments)
// ---------------------------------------------------------------------------

pub fn detect_faces(
    mat: &Mat,
    model_path: Option<&str>,
    input_size: Size,
    score_threshold: f32,
) -> Result<Mat> {
    let model_path = model_path.unwrap_or(DEFAULT_YUNET_MODEL_PATH);

    // Resize input for YuNet
    let mut resized = Mat::default();
    imgproc::resize(
        mat,
        &mut resized,
        input_size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    YUNET_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();

        // Create ONCE per thread
        if cache.is_none() {
            let detector = FaceDetectorYN::create(
                model_path,
                "",
                input_size,
                score_threshold,
                0.3,
                5000,
                0,
                0,
            )
            .map_err(|e| anyhow!("Failed to create YuNet detector: {}", e))?;

            *cache = Some(detector);
        }

        let detector = cache.as_mut().unwrap();

        // Update input size dynamically
        detector
            .set_input_size(input_size)
            .map_err(|e| anyhow!("Failed to set YuNet input size: {}", e))?;

        let mut faces = Mat::default();
        detector
            .detect(&resized, &mut faces)
            .map_err(|e| anyhow!("YuNet detection failed: {}", e))?;

        // Scale boxes back to original image
        let scale_x = mat.cols() as f32 / input_size.width as f32;
        let scale_y = mat.rows() as f32 / input_size.height as f32;

        for row in 0..faces.rows() {
            *faces.at_2d_mut::<f32>(row, 0)? *= scale_x;
            *faces.at_2d_mut::<f32>(row, 1)? *= scale_y;
            *faces.at_2d_mut::<f32>(row, 2)? *= scale_x;
            *faces.at_2d_mut::<f32>(row, 3)? *= scale_y;

            for &col in &[4, 6, 8, 10, 12] {
                *faces.at_2d_mut::<f32>(row, col)? *= scale_x;
                *faces.at_2d_mut::<f32>(row, col + 1)? *= scale_y;
            }
        }

        Ok(faces)
    })
}

pub fn pick_best_face(faces: &Mat) -> Result<Option<(Rect, Vec<Point2f>)>> {
    if faces.empty() || faces.rows() == 0 {
        return Ok(None);
    }

    let mut best_idx = -1;
    let mut best_score = -1.0_f32;

    for row in 0..faces.rows() {
        let score = *faces.at_2d::<f32>(row, 14)?; // score is column 14
        if score > best_score {
            best_score = score;
            best_idx = row;
        }
    }

    if best_idx < 0 {
        return Ok(None);
    }

    // Bounding box
    let x = *faces.at_2d::<f32>(best_idx, 0)? as i32;
    let y = *faces.at_2d::<f32>(best_idx, 1)? as i32;
    let w = *faces.at_2d::<f32>(best_idx, 2)? as i32;
    let h = *faces.at_2d::<f32>(best_idx, 3)? as i32;
    let bbox = Rect::new(x, y, w, h);

    // 5 landmarks (right eye, left eye, nose, right mouth, left mouth)
    let mut landmarks = Vec::with_capacity(5);
    for &col in &[4, 6, 8, 10, 12] {
        let px = *faces.at_2d::<f32>(best_idx, col)? as f32;
        let py = *faces.at_2d::<f32>(best_idx, col + 1)? as f32;
        landmarks.push(Point2f::new(px, py));
    }

    Ok(Some((bbox, landmarks)))
}
