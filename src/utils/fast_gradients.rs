use image::{GrayImage, ImageBuffer, Luma};
use imageproc::gradients::{horizontal_scharr, vertical_scharr};

type GradientProduct = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

pub fn compute_gradients(img: &GrayImage) -> GradientProduct {
    let grad_x = horizontal_scharr(img);
    let grad_y = vertical_scharr(img);
    (grad_x, grad_y)
}
