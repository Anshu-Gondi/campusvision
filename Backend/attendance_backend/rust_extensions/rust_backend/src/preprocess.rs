use anyhow::Result;
use opencv::{core::AlgorithmHint, imgcodecs, imgproc, prelude::*};
use tch::Tensor;

pub fn preprocess_image(image_bytes: &[u8]) -> Result<Tensor> {
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::from_slice(image_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(
        &mat,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let data = rgb.data_bytes()?.to_vec();
    let tensor = Tensor::from_slice(&data)
        .view([rgb.rows() as i64, rgb.cols() as i64, 3])
        .permute(&[2, 0, 1])
        .to_kind(tch::Kind::Float)
        / 255.0;

    Ok(tensor.unsqueeze(0))
}
