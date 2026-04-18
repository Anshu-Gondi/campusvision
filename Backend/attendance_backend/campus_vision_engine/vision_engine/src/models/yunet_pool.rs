use anyhow::{ anyhow, Result };
use opencv::{ core::{ Mat, Size }, objdetect::FaceDetectorYN, prelude::* };
use std::sync::{ mpsc, atomic::{ AtomicUsize, Ordering, AtomicBool } };
use tokio::sync::oneshot;
use std::thread;
use std::sync::Arc;

pub struct DetectRequest {
    pub image: Mat,
    pub respond_to: oneshot::Sender<Result<Mat>>,
}

pub struct YuNetPool {
    senders: Vec<mpsc::SyncSender<DetectRequest>>,
    counter: AtomicUsize,
    shutdown: Arc<AtomicBool>,        // ← Added for graceful shutdown
}

impl YuNetPool {
    pub fn new(model_path: &str, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let shutdown = Arc::new(AtomicBool::new(false));

        for _ in 0..workers {
            let (tx, rx) = mpsc::sync_channel(200);
            let shutdown_clone = Arc::clone(&shutdown);
            start_worker(rx, model_path.to_string(), shutdown_clone);
            senders.push(tx);
        }

        Self {
            senders,
            counter: AtomicUsize::new(0),
            shutdown,
        }
    }

    pub async fn detect(&self, image: &Mat) -> Result<Mat> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("YuNetPool is shutting down"));
        }

        let (tx, rx) = oneshot::channel();

        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.senders.len();

        self.senders[index]
            .send(DetectRequest {
                image: image.clone(),
                respond_to: tx,
            })
            .map_err(|e| anyhow!("YuNet send failed: {}", e))?;

        rx.await.map_err(|e| anyhow!("YuNet worker crashed: {}", e))?
    }

    // ← NEW: Shutdown method
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        println!("🛑 YuNetPool shutdown signal sent");
    }
}

fn start_worker(
    receiver: mpsc::Receiver<DetectRequest>, 
    model_path: String,
    shutdown: Arc<AtomicBool>
) {
    thread::spawn(move || {
        let mut detector = match FaceDetectorYN::create(
            &model_path,
            "",
            Size::new(320, 320),
            0.7,
            0.3,
            5000,
            0,
            0
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to create YuNet detector: {e}");
                return;
            }
        };

        while let Ok(req) = receiver.recv() {
            if shutdown.load(Ordering::Relaxed) {
                break;   // graceful exit
            }

            let result: Result<Mat> = (|| {
                let width = req.image.cols();
                let height = req.image.rows();

                if width <= 0 || height <= 0 {
                    return Err(anyhow!("Invalid image size"));
                }

                detector.set_input_size(Size::new(width, height))?;

                let mut faces = Mat::default();
                detector.detect(&req.image, &mut faces)?;

                Ok(faces)
            })();

            if req.respond_to.send(result).is_err() {
                eprintln!("YuNet: receiver dropped");
            }
        }

        println!("YuNet worker shutting down.");
    });
}