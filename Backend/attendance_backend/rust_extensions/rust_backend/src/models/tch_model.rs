use anyhow::Result;
use once_cell::sync::OnceCell;
use tch::{CModule, Tensor};

static FACE_MODEL: OnceCell<CModule> = OnceCell::new();
static EMOTION_MODEL: OnceCell<CModule> = OnceCell::new();

fn load_face_model() -> &'static CModule {
    FACE_MODEL.get_or_init(|| {
        // Ensure you have a TorchScript model (.pt) here.
        CModule::load("models/facenet.pt").expect("Failed to load face TorchScript model at models/facenet.pt")
    })
}

fn load_emotion_model() -> &'static CModule {
    EMOTION_MODEL.get_or_init(|| {
        CModule::load("models/emotion.pt").expect("Failed to load emotion TorchScript model at models/emotion.pt")
    })
}

pub fn run_face_model(tensor: &Tensor) -> Result<Vec<f32>> {
    // Expecting a [1, 3, H, W] float tensor
    let emb_tensor = load_face_model().forward_ts(&[tensor])?;
    let numel = emb_tensor.numel();
    let mut out = vec![0f32; numel as usize];
    emb_tensor.copy_data(&mut out, numel as usize);
    Ok(out)
}

pub fn run_emotion_model(tensor: &Tensor) -> Result<i64> {
    let output = load_emotion_model().forward_ts(&[tensor])?;
    Ok(output.argmax(1, false).int64_value(&[0]))
}
