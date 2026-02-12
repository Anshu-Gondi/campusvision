use anyhow::{ anyhow, Result };
use once_cell::sync::OnceCell;
use onnxruntime::{ environment::Environment, session::Session, GraphOptimizationLevel };
use ndarray::{ Array4, ArrayD };
use std::{ cell::RefCell, collections::HashMap };

// ==========================
// GLOBAL ORT ENVIRONMENT
// ==========================

static ORT_ENV: OnceCell<Environment> = OnceCell::new();

fn ort_env() -> &'static Environment {
    ORT_ENV.get_or_init(|| {
        Environment::builder()
            .with_name("ort_env")
            .build()
            .expect("Failed to create ORT environment")
    })
}

// ==========================
// THREAD LOCAL SESSION CACHE
// ==========================

thread_local! {
    static SESSION_CACHE: RefCell<HashMap<String, Session<'static>>> = RefCell::new(HashMap::new());
}

// ==========================
// LOAD OR GET SESSION
// ==========================

fn with_session<F, T>(model_path: &str, f: F) -> Result<T>
    where F: FnOnce(&mut Session<'static>) -> Result<T>
{
    let key = model_path.to_string();

    SESSION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if !cache.contains_key(&key) {
            let builder = ort_env()
                .new_session_builder()
                .map_err(|e| anyhow!("Failed to create session builder: {e}"))?;

            let builder = builder
                .with_number_threads(1)?
                .with_optimization_level(GraphOptimizationLevel::Basic)?;

            let static_path: &'static str = Box::leak(model_path.to_string().into_boxed_str());

            let session = builder
                .with_model_from_file(static_path)
                .map_err(|e| anyhow!("Failed to load model {model_path}: {e}"))?;

            cache.insert(key.clone(), session);
        }

        let session = cache.get_mut(&key).ok_or_else(|| anyhow!("Session not found"))?;

        f(session)
    })
}

// ==========================
// RUN FACE MODEL
// ==========================

pub fn run_face_model_onnx(input_array: &Array4<f32>, model_path: &str) -> Result<Vec<f32>> {
    with_session(model_path, |session| {
        let input_info = session.inputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("Model has no inputs"))?;

        let model_shape: Vec<Option<usize>> = input_info.dimensions
            .iter()
            .map(|d| d.map(|x| x as usize))
            .collect();

        let input_shape = input_array.shape();

        let array_to_use: ArrayD<f32> = match (model_shape.as_slice(), input_shape) {
            ([Some(_), Some(c), Some(_), Some(_)], [_, cc, _, _]) if *cc == *c => {
                input_array.clone().into_dyn()
            }
            ([Some(_), Some(_), Some(_), Some(c)], [_, _, _, cc]) if *cc == *c => {
                input_array.clone().into_dyn()
            }
            _ =>
                anyhow::bail!(
                    "Input shape/layout mismatch: model {:?}, input {:?}",
                    model_shape,
                    input_shape
                ),
        };

        let outputs = session.run(vec![array_to_use])?;

        let slice = outputs[0]
            .as_slice()
            .ok_or_else(|| anyhow!("ORT output tensor not contiguous"))?;

        Ok(slice.to_vec())
    })
}

// ==========================
// RUN EMOTION MODEL
// ==========================

pub fn run_emotion_model_onnx(input_array: &Array4<f32>, model_path: &str) -> Result<i64> {
    with_session(model_path, |session| {
        let outputs = session.run(vec![input_array.clone().into_dyn()])?;

        let scores: &[f32] = outputs[0]
            .as_slice()
            .ok_or_else(|| anyhow!("ORT output tensor not contiguous"))?;

        let (idx, _) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .ok_or_else(|| anyhow!("Empty output from emotion model"))?;

        Ok(idx as i64)
    })
}

// ==========================
// WARM-UP
// ==========================

pub fn warm_up_onnx_models(model_paths: &[&str]) -> Result<()> {
    tracing::info!("🔥 Warming up ONNX models");

    let dummy = Array4::<f32>::zeros((1, 112, 112, 3));

    for model_path in model_paths {
        tracing::info!("⚡ Warming up {}", model_path);
        let _ = run_face_model_onnx(&dummy, model_path)?;
    }

    tracing::info!("✅ ONNX models warmed");

    Ok(())
}
