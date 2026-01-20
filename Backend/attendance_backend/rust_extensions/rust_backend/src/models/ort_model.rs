use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use onnxruntime::{
    environment::Environment,
    session::Session,
    GraphOptimizationLevel,
};
use std::{cell::RefCell, path::Path};
use tch::Tensor;
use ndarray::Array4;

//==========================
// GLOBAL ENV
//==========================
static ORT_ENV: OnceCell<Environment> = OnceCell::new();

fn ort_env() -> &'static Environment {
    ORT_ENV.get_or_init(|| {
        Environment::builder()
            .with_name("ort_env")
            .build()
            .expect("Failed to create ORT environment")
    })
}

//==========================
// THREAD-LOCAL SESSIONS
//==========================
thread_local! {
    static FACE_SESSION: RefCell<Option<Session<'static>>> = RefCell::new(None);
    static EMOTION_SESSION: RefCell<Option<Session<'static>>> = RefCell::new(None);
}

//==========================
// FACE MODEL
//==========================
pub fn run_face_model_onnx(tensor: &Tensor) -> Result<Vec<f32>> {
    if tensor.size() != [1, 3, 160, 160] {
        return Err(anyhow!("Unexpected tensor shape: {:?}", tensor.size()));
    }

    // SAFE tensor extraction
    let tensor = tensor.contiguous();
    let numel = tensor.numel();
    let mut data = vec![0f32; numel as usize];
    tensor.copy_data(&mut data, numel);

    let array = Array4::from_shape_vec((1, 3, 160, 160), data)?;

    FACE_SESSION.with(|cell| {
        let mut slot = cell.borrow_mut();

        if slot.is_none() {
            let model_path: &'static Path =
                Box::leak(Box::new(Path::new("models/facenet.onnx").to_owned()));

            let session = ort_env()
                .new_session_builder()?
                .with_optimization_level(GraphOptimizationLevel::Basic)?
                .with_model_from_file(model_path)?;

            *slot = Some(session);
        }

        let session = slot.as_mut().unwrap();
        let outputs = session.run(vec![array])?;

        let slice: &[f32] = outputs[0]
            .as_slice()
            .ok_or_else(|| anyhow!("ORT output tensor not contiguous"))?;

        Ok(slice.to_vec())
    })
}

//==========================
// EMOTION MODEL
//==========================
pub fn run_emotion_model_onnx(tensor: &Tensor) -> Result<i64> {
    if tensor.size() != [1, 3, 160, 160] {
        return Err(anyhow!("Unexpected tensor shape: {:?}", tensor.size()));
    }

    let tensor = tensor.contiguous();
    let numel = tensor.numel();
    let mut data = vec![0f32; numel as usize];
    tensor.copy_data(&mut data, numel);

    let array = Array4::from_shape_vec((1, 3, 160, 160), data)?;

    EMOTION_SESSION.with(|cell| {
        let mut slot = cell.borrow_mut();

        if slot.is_none() {
            let model_path: &'static Path =
                Box::leak(Box::new(Path::new("models/emotion.onnx").to_owned()));

            let session = ort_env()
                .new_session_builder()?
                .with_optimization_level(GraphOptimizationLevel::Basic)?
                .with_model_from_file(model_path)?;

            *slot = Some(session);
        }

        let session = slot.as_mut().unwrap();
        let outputs = session.run(vec![array])?;

        let scores: &[f32] = outputs[0]
            .as_slice()
            .ok_or_else(|| anyhow!("ORT output tensor not contiguous"))?;

        let (idx, _) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Ok(idx as i64)
    })
}
