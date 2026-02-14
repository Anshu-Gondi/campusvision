use anyhow::{anyhow, Result};
use ndarray::Array4;
use once_cell::sync::OnceCell;
use onnxruntime::{environment::Environment, GraphOptimizationLevel};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

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
// INFERENCE REQUEST
// ==========================

struct InferenceRequest {
    input: Array4<f32>,
    respond_to: oneshot::Sender<Result<Vec<f32>>>,
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
        self.total_latency_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn average_latency_ms(&self) -> f64 {
        let reqs = self.total_requests.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        (total_ns as f64 / reqs as f64) / 1_000_000.0
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
    ready: Arc<AtomicBool>, // 🔥 NEW
}

impl InferencePool {
    pub fn new(model_path: &str, workers: usize) -> Self {
        assert!(workers > 0, "workers must be > 0");

        let mut senders = Vec::with_capacity(workers);
        let mut metrics = Vec::with_capacity(workers);
        let shutdown = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false)); // 🔥 NEW

        for worker_id in 0..workers {
            let (tx, rx) = mpsc::sync_channel(50);
            let worker_metrics = Arc::new(WorkerMetrics::default());

            start_worker(
                worker_id,
                rx,
                model_path.to_string(),
                shutdown.clone(),
                worker_metrics.clone(),
            );

            senders.push(tx);
            metrics.push(worker_metrics);
        }

        // 🔥 Mark pool ready after workers spawned
        ready.store(true, Ordering::Relaxed);

        Self {
            senders,
            counter: AtomicUsize::new(0),
            shutdown,
            metrics,
            ready,
        }
    }

    // 🔥 Async inference entrypoint (BACKPRESSURE SAFE)
    pub async fn infer(&self, input: Array4<f32>) -> Result<Vec<f32>> {
        if !self.is_ready() {
            return Err(anyhow!("InferencePool not ready"));
        }

        let (tx, rx) = oneshot::channel();

        let index =
            self.counter.fetch_add(1, Ordering::Relaxed) % self.senders.len();

        // ✅ BLOCKING send (correct for SaaS backpressure)
        self.senders[index]
            .send(InferenceRequest {
                input,
                respond_to: tx,
            })
            .map_err(|_| anyhow!("Inference queue closed"))?;

        rx.await.map_err(|_| anyhow!("Worker crashed"))?
    }

    // 🔥 Warm up ALL workers
    pub fn warm_up(&self, dummy: Array4<f32>) {
        for sender in &self.senders {
            let _ = sender.send(InferenceRequest {
                input: dummy.clone(),
                respond_to: oneshot::channel().0,
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
) {
    thread::spawn(move || {
        #[cfg(target_os = "linux")]
        {
            if let Some(cores) = core_affinity::get_core_ids() {
                if let Some(core) = cores.get(worker_id % cores.len()) {
                    core_affinity::set_for_current(*core);
                }
            }
        }

        let env = ort_env();

        let session = env
            .new_session_builder()
            .expect("Failed builder")
            .with_number_threads(1)
            .expect("Thread config failed")
            .with_optimization_level(GraphOptimizationLevel::All)
            .expect("Opt level failed")
            .with_model_from_file(&model_path)
            .expect("Model load failed");

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(req) => {
                    let start = Instant::now();

                    let result = session
                        .run(vec![req.input.into_dyn()])
                        .map_err(|e| anyhow!("{e}"))
                        .and_then(|outputs| {
                            let slice = outputs[0]
                                .as_slice()
                                .ok_or_else(|| anyhow!("Output not contiguous"))?;
                            Ok(slice.to_vec())
                        });

                    metrics.record(start.elapsed());

                    let _ = req.respond_to.send(result);
                }

                Err(mpsc::RecvTimeoutError::Timeout) => continue,

                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        println!("Worker {} shutting down.", worker_id);
    });
}
