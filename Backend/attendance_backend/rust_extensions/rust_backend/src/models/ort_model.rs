use anyhow::Result;
use once_cell::sync::OnceCell;
use onnxruntime::{
    environment::Environment,
    session::Session,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use std::{
    path::Path,
    sync::{Mutex, MutexGuard},
    cell::UnsafeCell,
};
use tch::Tensor;

//==========================
// GLOBAL ORT ENVIRONMENT
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

//================================================
// SESSION POOL: Fully Parallel ONNX Inference
//================================================

pub struct SessionPool {
    sessions: Vec<Mutex<UnsafeSession>>,
}

// Wrapper to make Session Send + Sync safely through Mutex
pub struct UnsafeSession(UnsafeCell<Session<'static>>);

// SAFETY: We ensure thread-safe access via Mutex
unsafe impl Send for UnsafeSession {}
unsafe impl Sync for UnsafeSession {}

impl SessionPool {
    pub fn new(pool_size: usize, model_path: impl AsRef<Path>) -> Self {
        // Leak the path to get a &'static Path for Session<'static>
        let model_path: &'static Path = Box::leak(Box::new(model_path.as_ref().to_owned()));

        let mut sessions = Vec::with_capacity(pool_size);

        for _ in 0..pool_size {
            // Create a fresh SessionBuilder for each session
            let builder = ort_env()
                .new_session_builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Basic)
                .unwrap();

            // Build session
            let session = builder
                .with_model_from_file(model_path)
                .expect("Failed to load ONNX model");

            sessions.push(Mutex::new(UnsafeSession(UnsafeCell::new(session))));
        }

        Self { sessions }
    }

    pub fn acquire(&'static self) -> SessionGuard {
        // Try to lock any available session first
        for (idx, m) in self.sessions.iter().enumerate() {
            if let Ok(guard) = m.try_lock() {
                return SessionGuard {
                    pool: self,
                    index: idx,
                    session_guard: Some(guard),
                };
            }
        }

        // If all are locked, wait on the first one
        let guard = self.sessions[0].lock().unwrap();
        SessionGuard {
            pool: self,
            index: 0,
            session_guard: Some(guard),
        }
    }

    fn release(&self, _index: usize) {
        // MutexGuard drop automatically unlocks, nothing else needed
    }
}

pub struct SessionGuard {
    pool: &'static SessionPool,
    index: usize,
    session_guard: Option<MutexGuard<'static, UnsafeSession>>,
}

impl SessionGuard {
    pub fn session(&mut self) -> &mut Session<'static> {
        unsafe { &mut *self.session_guard.as_mut().unwrap().0.get() }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.session_guard.take();
        self.pool.release(self.index);
    }
}

//================================================
// FACE POOL & EMOTION POOL (GLOBAL STATIC)
//================================================

static FACE_POOL: OnceCell<SessionPool> = OnceCell::new();
static EMOTION_POOL: OnceCell<SessionPool> = OnceCell::new();

fn face_pool() -> &'static SessionPool {
    FACE_POOL.get_or_init(|| SessionPool::new(6, "models/facenet.onnx"))
}

fn emotion_pool() -> &'static SessionPool {
    EMOTION_POOL.get_or_init(|| SessionPool::new(4, "models/emotion.onnx"))
}

//=====================================
//     MODEL EXECUTION HELPERS
//=====================================

pub fn run_face_model_onnx(tensor: &Tensor) -> Result<Vec<f32>> {
    let data: Vec<f32> = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view(-1)
        .try_into()
        .expect("Failed to flatten tensor");

    let mut guard = face_pool().acquire();
    let session = guard.session();

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![data.into()])?;
    Ok(outputs[0].as_slice().unwrap().to_vec())
}

pub fn run_emotion_model_onnx(tensor: &Tensor) -> Result<i64> {
    let data: Vec<f32> = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view(-1)
        .try_into()
        .expect("Failed to flatten tensor");

    let mut guard = emotion_pool().acquire();
    let session = guard.session();

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![data.into()])?;
    let scores = outputs[0].as_slice().unwrap();

    let (max_idx, _) =
        scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

    Ok(max_idx as i64)
}
