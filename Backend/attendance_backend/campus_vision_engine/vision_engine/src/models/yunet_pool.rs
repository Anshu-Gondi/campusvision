use anyhow::{ anyhow, Result };
use opencv::{ core::{ Mat, Size }, objdetect::FaceDetectorYN, prelude::* };
use std::sync::{ mpsc, atomic::{ AtomicUsize, Ordering } };
use tokio::sync::oneshot;
use std::thread;

pub struct DetectRequest {
    pub image: Mat,
    pub respond_to: oneshot::Sender<Result<Mat>>,
}

pub struct YuNetPool {
    senders: Vec<mpsc::SyncSender<DetectRequest>>,
    counter: AtomicUsize,
}

impl YuNetPool {
    pub fn new(model_path: &str, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);

        for _ in 0..workers {
            let (tx, rx) = mpsc::sync_channel(200); // backpressure: 1 request per worker
            start_worker(rx, model_path.to_string());
            senders.push(tx);
        }

        Self {
            senders,
            counter: AtomicUsize::new(0),
        }
    }

    pub async fn detect(&self, image: &Mat) -> Result<Mat> {
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
}

fn start_worker(receiver: mpsc::Receiver<DetectRequest>, model_path: String) {
    thread::spawn(move || {
        let mut detector = match
            FaceDetectorYN::create(
                &model_path,
                "",
                Size::new(320, 320), // initial dummy
                0.7,
                0.3,
                5000,
                0,
                0
            )
        {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to create YuNet detector: {e}");
                return;
            }
        };

        for req in receiver {
            let result: Result<Mat> = (|| {
                let width = req.image.cols();
                let height = req.image.rows();

                if width <= 0 || height <= 0 {
                    return Err(anyhow!("Invalid image size"));
                }

                // 🔥 YuNet requires setting input size per image
                detector.set_input_size(Size::new(width, height))?;

                let mut faces = Mat::default();
                detector.detect(&req.image, &mut faces)?;

                Ok(faces)
            })();

            if req.respond_to.send(result).is_err() {
                eprintln!("YuNet: receiver dropped");
            }
        }
    });
}
