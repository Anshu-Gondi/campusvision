// src/models/onnx_models.rs

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use onnxruntime::{
    environment::Environment,
    session::Session,
    GraphOptimizationLevel,
};
use ndarray::Array4;
use std::{cell::RefCell, path::Path};

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
pub fn run_face_model_onnx(input_array: &Array4<f32>) -> Result<Vec<f32>> {
    if input_array.shape() != [1, 3, 160, 160] {
        return Err(anyhow!("Unexpected input array shape: {:?}", input_array.shape()));
    }

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
        let outputs = session.run(vec![input_array.clone().into_dyn()])?;

        let slice: &[f32] = outputs[0]
            .as_slice()
            .ok_or_else(|| anyhow!("ORT output tensor not contiguous"))?;

        Ok(slice.to_vec())
    })
}

//==========================
// EMOTION MODEL
//==========================
pub fn run_emotion_model_onnx(input_array: &Array4<f32>) -> Result<i64> {
    if input_array.shape() != [1, 3, 160, 160] {
        return Err(anyhow!("Unexpected input array shape: {:?}", input_array.shape()));
    }

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
        let outputs = session.run(vec![input_array.clone().into_dyn()])?;

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
