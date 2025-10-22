use opencv::{core::AlgorithmHint, imgcodecs, imgproc, prelude::*};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::OnceLock;
use tch::{nn, Device, Tensor};

// Global ONNX model instance
static FACE_MODEL: OnceLock<tch::CModule> = OnceLock::new();
static EMOTION_MODEL: OnceLock<tch::CModule> = OnceLock::new();

fn load_face_model() -> &'static tch::CModule {
    FACE_MODEL.get_or_init(|| {
        tch::CModule::load("models/facenet.onnx").expect("Failed to load face model")
    })
}

fn load_emotion_model() -> &'static tch::CModule {
    EMOTION_MODEL.get_or_init(|| {
        tch::CModule::load("models/emotion.onnx").expect("Failed to load emotion model")
    })
}

/// Convert image bytes to RGB tensor
fn preprocess_image(image_bytes: &[u8]) -> anyhow::Result<Tensor> {
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::from_slice(image_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;
    let mut rgb = Mat::default();
    imgproc::cvt_color(&mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let data: Vec<u8> = rgb.data_bytes()?.to_vec();

    let tensor = Tensor::f_from_slice(&data)?; // fallible
    let tensor = tensor
        .view([rgb.rows() as i64, rgb.cols() as i64, 3])
        .permute(&[2, 0, 1])
        .to_kind(tch::Kind::Float)
        / 255.0;
    let tensor = tensor.unsqueeze(0);

    Ok(tensor) // ✅ return the tensor wrapped in Result
}

/// Face verification: cosine similarity
#[pyfunction]
fn verify_face(input_image: Vec<u8>, known_embedding: Vec<f32>) -> PyResult<f32> {
    let img_tensor = preprocess_image(&input_image)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let model = load_face_model();
    let emb_tensor = model
        .forward_ts(&[img_tensor])
        .map_err(|e| PyValueError::new_err(e.to_string()))?; // [1, 512]

    // Allocate a Vec with the correct size
    let numel = emb_tensor.numel();
    let mut emb_vec = vec![0f32; numel as usize];

    // Copy tensor data into the Vec
    emb_tensor.f_contiguous()
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .f_copy_data(&mut emb_vec, numel as usize)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Cosine similarity
    let dot = emb_vec
        .iter()
        .zip(known_embedding.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    let norm_a = emb_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = known_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    Ok(dot / (norm_a * norm_b))
}

/// Emotion detection: returns dominant emotion index
#[pyfunction]
fn detect_emotion(input_image: Vec<u8>) -> PyResult<i64> {
    let img_tensor =
        preprocess_image(&input_image).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let model = load_emotion_model();
    let output = model.forward_ts(&[img_tensor]).unwrap(); // [1, num_emotions]
    let argmax = output.argmax(1, false);
    Ok(argmax.int64_value(&[0]))
}

#[pymodule]
fn rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_emotion, m)?)?;
    Ok(())
}
