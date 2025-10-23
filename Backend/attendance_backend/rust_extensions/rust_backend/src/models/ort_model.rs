use anyhow::Result;
use once_cell::sync::OnceCell;
use onnxruntime::{
    environment::Environment,
    session::Session,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use std::sync::Mutex;
use tch::Tensor;

/// Unsafe wrapper to allow static storage of ONNX sessions.
/// We promise Rust we'll never use them across threads unsafely.
struct SafeSession {
    inner: Mutex<Session<'static>>,
}

// We guarantee safety by only accessing sessions via Mutex (single-threaded).
unsafe impl Send for SafeSession {}
unsafe impl Sync for SafeSession {}

static ORT_ENV: OnceCell<Environment> = OnceCell::new();
static FACE_SESSION: OnceCell<SafeSession> = OnceCell::new();
static EMOTION_SESSION: OnceCell<SafeSession> = OnceCell::new();

fn ort_env() -> &'static Environment {
    ORT_ENV.get_or_init(|| {
        Environment::builder()
            .with_name("ort_env")
            .build()
            .expect("Failed to create ORT environment")
    })
}

fn face_session() -> &'static SafeSession {
    FACE_SESSION.get_or_init(|| {
        let session = ort_env()
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_model_from_file("models/facenet.onnx")
            .unwrap();
        SafeSession {
            inner: Mutex::new(session),
        }
    })
}

fn emotion_session() -> &'static SafeSession {
    EMOTION_SESSION.get_or_init(|| {
        let session = ort_env()
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_model_from_file("models/emotion.onnx")
            .unwrap();
        SafeSession {
            inner: Mutex::new(session),
        }
    })
}

pub fn run_face_model_onnx(tensor: &Tensor) -> Result<Vec<f32>> {
    let data: Vec<f32> = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view(-1)
        .try_into()
        .expect("Failed to convert tensor to Vec<f32>");

    let mut session = face_session().inner.lock().unwrap();
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![data.into()])?;
    Ok(outputs[0].as_slice().unwrap().to_vec())
}

pub fn run_emotion_model_onnx(tensor: &Tensor) -> Result<i64> {
    let data: Vec<f32> = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view(-1)
        .try_into()
        .expect("Failed to convert tensor to Vec<f32>");

    let mut session = emotion_session().inner.lock().unwrap();
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![data.into()])?;
    let scores = outputs[0].as_slice().unwrap();

    let (max_idx, _) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    Ok(max_idx as i64)
}
