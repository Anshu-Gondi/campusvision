/// models/onnx_models.rs
///
/// Replaces ORT-based InferencePool with tract-onnx.
///
/// tract is pure Rust — no C++ runtime, no DLL, no AVX requirement.
/// It uses whatever SIMD the CPU supports (SSE2 minimum, SSE4.1 if present)
/// detected at compile time via rustc's target-feature flags.
///
/// API is identical to the old InferencePool so no other files change.

use anyhow::{ anyhow, Result };
use ndarray::Array4;
use once_cell::sync::OnceCell;
use std::sync::{ atomic::{ AtomicBool, AtomicU64, AtomicUsize, Ordering }, mpsc, Arc };
use std::thread::{ self, JoinHandle };
use std::time::{ Duration, Instant };
use tokio::sync::oneshot;
use tract_ndarray::ArrayD;
use tract_onnx::prelude::*;

// ── Type alias for the compiled tract model ───────────────────────────────────
//
// SimplePlan is tract's optimized execution plan.
// TypedFact + TypedOp = fully type-inferred model (fastest execution path).
type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ── Normalize embedding (unchanged from original) ─────────────────────────────

fn normalize_embedding(v: &mut [f32]) {
    let chunks = v.chunks_exact(4);
    let remainder = chunks.remainder();
    let mut sum = 0.0f32;

    for chunk in chunks {
        sum +=
            chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3];
    }
    for &x in remainder {
        sum += x * x;
    }

    let norm = sum.sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

// ── Inference request (unchanged shape) ──────────────────────────────────────

struct InferenceRequest {
    input: Array4<f32>,
    respond_to: oneshot::Sender<Vec<Vec<f32>>>,
}

// ── Metrics (unchanged) ───────────────────────────────────────────────────────

#[derive(Default)]
pub struct WorkerMetrics {
    pub total_requests: AtomicUsize,
    pub total_latency_ns: AtomicU64,
}

impl WorkerMetrics {
    pub fn record(&self, duration: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn average_latency_ms(&self) -> f64 {
        let reqs = self.total_requests.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        let ns = self.total_latency_ns.load(Ordering::Relaxed);
        (ns as f64) / (reqs as f64) / 1_000_000.0
    }
}

// ── InferencePool (same public API as before) ─────────────────────────────────

pub struct InferencePool {
    senders: Vec<mpsc::SyncSender<InferenceRequest>>,
    counter: AtomicUsize,
    shutdown: Arc<AtomicBool>,
    metrics: Vec<Arc<WorkerMetrics>>,
    ready: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>,
}

impl InferencePool {
    pub fn new(model_path: &str, workers: usize) -> Self {
        assert!(workers > 0, "workers must be > 0");

        // Determine correct input shape based on model name
        let input_shape: Vec<usize> = if model_path.to_lowercase().contains("emotion") 
            || model_path.to_lowercase().contains("ferplus") {
            vec![1, 1, 64, 64]      // NCHW grayscale for Emotion FERPlus
        } else {
            vec![1, 112, 112, 3]    // NHWC RGB for ArcFace
        };

        // Load model with correct shape
        let model = load_model(model_path, &input_shape)
            .unwrap_or_else(|e| {
                panic!("Failed to load model {}: {}", model_path, e);
            });

        let model = Arc::new(model);

        // ... rest of the function remains the same ...
        let mut senders = Vec::with_capacity(workers);
        let mut metrics = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);

        let shutdown = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false));
        let ready_counter = Arc::new(AtomicUsize::new(0));

        for worker_id in 0..workers {
            let (tx, rx) = mpsc::sync_channel::<InferenceRequest>(50);
            let worker_metrics = Arc::new(WorkerMetrics::default());

            let handle = start_worker(
                worker_id,
                rx,
                Arc::clone(&model),
                Arc::clone(&shutdown),
                Arc::clone(&worker_metrics),
                Arc::clone(&ready),
                Arc::clone(&ready_counter),
                workers
            );

            senders.push(tx);
            metrics.push(worker_metrics);
            handles.push(handle);
        }

        Self {
            senders,
            counter: AtomicUsize::new(0),
            shutdown,
            metrics,
            ready,
            _handles: handles,
        }
    }

    // ── Public API (identical to old ORT version) ─────────────────────────

    pub async fn infer(&self, input: Array4<f32>) -> Result<Vec<f32>> {
        if !self.is_ready() {
            return Err(anyhow!("InferencePool not ready"));
        }

        let (tx, rx) = oneshot::channel();
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.senders.len();

        self.senders[index]
            .send(InferenceRequest {
                input,
                respond_to: tx,
            })
            .map_err(|_| anyhow!("Inference queue closed"))?;

        let mut result = rx.await.map_err(|_| anyhow!("Worker crashed"))?;

        if result.is_empty() {
            return Err(anyhow!("Empty output from model"));
        }

        Ok(result.remove(0))
    }

    pub async fn infer_batch(&self, batch: Vec<Array4<f32>>) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = batch.len();
        let shape = batch[0].shape().to_vec();

        // Combine into single NCHW tensor
        let mut combined = Array4::<f32>::zeros((batch_size, shape[1], shape[2], shape[3]));
        for (i, input) in batch.into_iter().enumerate() {
            combined.slice_mut(ndarray::s![i, .., .., ..]).assign(&input);
        }

        let (tx, rx) = oneshot::channel();
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.senders.len();

        self.senders[index]
            .send(InferenceRequest {
                input: combined,
                respond_to: tx,
            })
            .map_err(|e| anyhow!("Batch send failed: {}", e))?;

        rx.await.map_err(|e| anyhow!("Worker crashed: {}", e))
    }

    pub async fn run_face_embedding(&self, input: Array4<f32>) -> Result<Vec<f32>> {
        self.infer(input).await
    }

    pub async fn run_emotion(&self, input: Array4<f32>) -> Result<i64> {
        let output = self.infer(input).await?;
        Ok(
            output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i64)
                .unwrap_or(0)
        )
    }

    pub fn get_model_input_size(&self) -> Result<(usize, usize)> {
        Ok((112, 112))
    }

    pub fn warm_up(&self, dummy: Array4<f32>) {
        // Send one request per worker to trigger JIT compilation inside tract.
        // tract compiles the execution plan lazily on first run.
        for sender in &self.senders {
            let (tx, _rx) = oneshot::channel();
            let _ = sender.send(InferenceRequest {
                input: dummy.clone(),
                respond_to: tx,
            });
        }
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    pub fn print_metrics(&self) {
        for (i, m) in self.metrics.iter().enumerate() {
            println!(
                "Worker {} → avg latency: {:.2} ms | total reqs: {}",
                i,
                m.average_latency_ms(),
                m.total_requests.load(Ordering::Relaxed)
            );
        }
    }
}

// ── Model loading ─────────────────────────────────────────────────────────────
//
// tract loads the ONNX file, infers all types, and optimizes the graph
// into a TypedModel. This is the slow step — done once at startup.
// The resulting SimplePlan is fast and allocation-minimal at inference time.

fn load_model(model_path: &str, input_shape: &[usize]) -> Result<TractModel> {
    println!("Loading model: {} with input shape {:?}", model_path, input_shape);

    let shape = tvec!(
        input_shape[0] as i64,   // batch
        input_shape[1] as i64,
        input_shape[2] as i64,
        input_shape[3] as i64
    );

    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), shape)
        )?
        .into_optimized()?
        .into_runnable()?;

    println!("✅ Successfully optimized model: {}", model_path);
    Ok(model)
}

// ── Worker thread ─────────────────────────────────────────────────────────────
//
// Each worker holds a clone of the Arc<TractModel>.
// tract's SimplePlan is designed to be used from one thread at a time
// (it has internal mutable state for execution buffers).
// That is exactly our model: one worker thread per sender.

fn start_worker(
    worker_id: usize,
    receiver: mpsc::Receiver<InferenceRequest>,
    model: Arc<TractModel>,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<WorkerMetrics>,
    ready: Arc<AtomicBool>,
    ready_counter: Arc<AtomicUsize>,
    total_workers: usize
) -> JoinHandle<()> {
    thread::spawn(move || {
        // Pin to CPU core on Linux (same as before)
        #[cfg(target_os = "linux")]
        {
            if let Some(cores) = core_affinity::get_core_ids() {
                if let Some(core) = cores.get(worker_id % cores.len()) {
                    core_affinity::set_for_current(*core);
                }
            }
        }

        // Mark ready immediately — model is already loaded (no per-worker load)
        if ready_counter.fetch_add(1, Ordering::SeqCst) + 1 == total_workers {
            ready.store(true, Ordering::SeqCst);
        }

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(req) => {
                    let start = Instant::now();

                    let result = run_inference(&model, req.input);

                    metrics.record(start.elapsed());

                    match result {
                        Ok(embeddings) => {
                            let _ = req.respond_to.send(embeddings);
                        }
                        Err(e) => {
                            eprintln!("Inference error worker {}: {:?}", worker_id, e);
                            let _ = req.respond_to.send(vec![]);
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }

        println!("Worker {} shutting down.", worker_id);
    })
}

// ── Core inference function ───────────────────────────────────────────────────

fn run_inference(model: &TractModel, input: Array4<f32>) -> Result<Vec<Vec<f32>>> {
    // Detect layout from shape:
    // - ArcFace:  shape = [1, 112, 112, 3]  → NHWC
    // - Emotion:  shape = [1, 1, 64, 64]    → NCHW
    let shape = input.shape().to_vec();           // clone shape first
    let is_nhwc = shape.len() == 4 && shape[3] == 3;

    let tract_input = if is_nhwc {
        // ArcFace: NHWC → convert to NCHW
        let nchw = input.permuted_axes([0, 3, 1, 2]);
        let data = nchw.into_raw_vec();
        Tensor::from_shape(&shape[0..4], &data)?     // use original cloned shape (but correct order)
    } else {
        // Emotion: already NCHW, use as-is
        let data = input.into_raw_vec();
        Tensor::from_shape(&shape, &data)?
    };

    // Run inference
    let outputs = model.run(tvec![tract_input.into()])?;

    let output_tensor = outputs[0]
        .to_array_view::<f32>()
        .map_err(|e| anyhow!("Output extraction failed: {}", e))?;

    let batch_size = output_tensor.shape()[0];
    let feature_dim = output_tensor.len() / batch_size;

    let flat: Vec<f32> = output_tensor.iter().copied().collect();
    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let start = i * feature_dim;
        let end = start + feature_dim;
        let mut v = flat[start..end].to_vec();

        // Only normalize if it's a high-dimensional embedding (ArcFace ~512 dim)
        // Emotion outputs only 8 values → do NOT normalize
        if v.len() > 20 {
            normalize_embedding(&mut v);
        }

        results.push(v);
    }

    Ok(results)
}