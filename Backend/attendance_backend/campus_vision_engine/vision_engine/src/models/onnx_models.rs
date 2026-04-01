use anyhow::{ anyhow, Result };
use ndarray::Array4;
use once_cell::sync::OnceCell;
use ort::{
    environment::Environment,
    session::builder::{ GraphOptimizationLevel, SessionBuilder },
    value::Value,
};
use std::sync::{ atomic::{ AtomicBool, AtomicU64, AtomicUsize, Ordering }, mpsc, Arc };
use std::thread::{ self, JoinHandle };
use std::time::{ Duration, Instant };
use tokio::sync::oneshot;

// ==========================
// UTILITIES
// ==========================

fn normalize_embedding(v: &mut [f32]) {
    let mut sum = 0.0;

    // manual unroll (cheap SIMD gain)
    let chunks = v.chunks_exact(4);
    let remainder = chunks.remainder();

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

// ==========================
// INFERENCE REQUEST
// ==========================

struct InferenceRequest {
    input: Array4<f32>,
    respond_to: oneshot::Sender<Vec<Vec<f32>>>,
}

// ==========================
// METRICS
// ==========================

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
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        (total_ns as f64) / (reqs as f64) / 1_000_000.0
    }
}

// ==========================
// INFERENCE POOL
// ==========================

pub struct InferencePool {
    senders: Vec<mpsc::SyncSender<InferenceRequest>>,
    counter: AtomicUsize,
    shutdown: Arc<AtomicBool>,
    metrics: Vec<Arc<WorkerMetrics>>,
    ready: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>, // keep threads alive
}

impl InferencePool {
    pub fn new(model_path: &str, workers: usize) -> Self {
        assert!(workers > 0, "workers must be > 0");

        let mut senders = Vec::with_capacity(workers);
        let mut metrics = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);

        let shutdown = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false));
        let ready_counter = Arc::new(AtomicUsize::new(0));

        for worker_id in 0..workers {
            let (tx, rx) = mpsc::sync_channel(50);
            let worker_metrics = Arc::new(WorkerMetrics::default());

            let handle = start_worker(
                worker_id,
                rx,
                model_path.to_string(),
                shutdown.clone(),
                worker_metrics.clone(),
                ready.clone(),
                ready_counter.clone(),
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

    pub async fn infer_batch(&self, batch: Vec<Array4<f32>>) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let (tx, rx) = oneshot::channel();

        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.senders.len();

        // 🔥 combine batch into single tensor
        let batch_size = batch.len();
        let shape = batch[0].shape();

        let mut combined = Array4::<f32>::zeros((batch_size, shape[1], shape[2], shape[3]));

        for (i, input) in batch.into_iter().enumerate() {
            combined.slice_mut(ndarray::s![i, .., .., ..]).assign(&input);
        }

        self.senders[index]
            .send(InferenceRequest {
                input: combined,
                respond_to: tx,
            })
            .map_err(|e| anyhow!("Batch send failed: {}", e))?;

        let flat = rx.await.map_err(|e| anyhow!("Worker crashed: {}", e))?;

        Ok(flat)
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
        Ok((112, 112)) // or your real model size
    }

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
            return Err(anyhow!("Empty output"));
        }

        Ok(result.remove(0))
    }

    pub fn warm_up(&self, dummy: Array4<f32>) {
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

// ==========================
// WORKER THREAD
// ==========================

fn start_worker(
    worker_id: usize,
    receiver: mpsc::Receiver<InferenceRequest>,
    model_path: String,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<WorkerMetrics>,
    ready: Arc<AtomicBool>,
    ready_counter: Arc<AtomicUsize>,
    total_workers: usize
) -> JoinHandle<()> {
    thread::spawn(move || {
        #[cfg(target_os = "linux")]
        {
            if let Some(cores) = core_affinity::get_core_ids() {
                if let Some(core) = cores.get(worker_id % cores.len()) {
                    core_affinity::set_for_current(*core);
                }
            }
        }

        let mut session = SessionBuilder::new()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(1) // keep 1 per worker
            .unwrap()
            .with_parallel_execution(false)
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();

        // mark worker ready AFTER model loaded
        if ready_counter.fetch_add(1, Ordering::SeqCst) + 1 == total_workers {
            ready.store(true, Ordering::SeqCst);
        }

        let mut input_buffer: Option<ndarray::ArrayD<f32>> = None;

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(req) => {
                    let start = Instant::now();

                    let input = req.input;

                    let input_dyn = if let Some(ref mut buf) = input_buffer {
                        buf.assign(&input);
                        buf
                    } else {
                        input_buffer = Some(req.input.into_dyn());
                        input_buffer.as_mut().unwrap()
                    };

                    let result = (|| -> Result<Vec<Vec<f32>>> {
                        // 🔥 Flatten input (NO clone of ndarray needed)
                        let shape = input.shape().to_vec();
                        let data = input.into_raw_vec();

                        // Create ORT tensor
                        let input_tensor = Value::from_array((shape.clone(), data))?;

                        // ⚠️ IMPORTANT: replace "input" with your real model input name
                        let outputs = session.run(vec![("input", input_tensor)])?;

                        let output = &outputs[0];

                        // 🔥 SAFEST extraction for this ORT version
                        let (shape, data) = output.try_extract_tensor::<f32>()?;

                        let batch = shape[0] as usize;
                        let dim = shape[1] as usize;

                        let mut results = Vec::with_capacity(batch);

                        for i in 0..batch {
                            let start = i * dim;
                            let end = start + dim;

                            let mut v = data[start..end].to_vec();
                            normalize_embedding(&mut v);

                            results.push(v);
                        }

                        Ok(results)
                    })();

                    metrics.record(start.elapsed());

                    match result {
                        Ok(res) => {
                            let _ = req.respond_to.send(res);
                        }
                        Err(e) => {
                            eprintln!("Inference error: {:?}", e);
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
