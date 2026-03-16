use std::hint::black_box;
use std::time::{Duration, Instant};

use image::{GrayImage, ImageBuffer, Luma};
use imageproc::gradients::{gradients_grayscale, horizontal_scharr, vertical_scharr};
use imageproc::kernel::{SCHARR_HORIZONTAL_3X3, SCHARR_VERTICAL_3X3};

const HORIZONTAL_SCHARR_3X3_OLD: [i32; 9] = [-3, 0, 3, -10, 0, 10, -3, 0, 3];
const VERTICAL_SCHARR_3X3_OLD: [i32; 9] = [-3, -10, -3, 0, 0, 0, 3, 10, 3];

type GradientPair = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

fn manual_scharr_old(img: &GrayImage, kernel_x: &[i32; 9], kernel_y: &[i32; 9]) -> GradientPair {
    let (width, height) = img.dimensions();
    let mut grad_x = ImageBuffer::new(width, height);
    let mut grad_y = ImageBuffer::new(width, height);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = img.get_pixel(x + kx - 1, y + ky - 1)[0] as i32;
                    gx += pixel * kernel_x[(ky * 3 + kx) as usize];
                    gy += pixel * kernel_y[(ky * 3 + kx) as usize];
                }
            }

            grad_x.put_pixel(x, y, Luma([gx as i16]));
            grad_y.put_pixel(x, y, Luma([gy as i16]));
        }
    }

    (grad_x, grad_y)
}

fn imageproc_scharr_pair(img: &GrayImage) -> GradientPair {
    (horizontal_scharr(img), vertical_scharr(img))
}

fn imageproc_scharr_magnitude(img: &GrayImage) -> ImageBuffer<Luma<u16>, Vec<u16>> {
    gradients_grayscale(img, SCHARR_HORIZONTAL_3X3, SCHARR_VERTICAL_3X3)
}

fn make_test_image(width: u32, height: u32) -> GrayImage {
    let mut img = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let v = ((x * 31 + y * 17 + (x ^ y) * 13) & 0xff) as u8;
            img.put_pixel(x, y, Luma([v]));
        }
    }

    img
}

fn time_it<T, F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut() -> T,
{
    for _ in 0..3 {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());
    }
    start.elapsed()
}

fn print_result(label: &str, iterations: usize, elapsed: Duration) {
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iter_ms = total_ms / iterations as f64;
    println!("{label:<32} total: {total_ms:>10.3} ms   per-iter: {per_iter_ms:>8.3} ms");
}

fn run_case(width: u32, height: u32, iterations: usize) {
    let img = make_test_image(width, height);

    println!();
    println!("Image: {width}x{height}  iterations: {iterations}");

    let old_elapsed = time_it(iterations, || {
        manual_scharr_old(
            &img,
            black_box(&HORIZONTAL_SCHARR_3X3_OLD),
            black_box(&VERTICAL_SCHARR_3X3_OLD),
        )
    });
    print_result("manual_scharr_old", iterations, old_elapsed);

    let pair_elapsed = time_it(iterations, || imageproc_scharr_pair(&img));
    print_result("imageproc_scharr_pair", iterations, pair_elapsed);

    let mag_elapsed = time_it(iterations, || imageproc_scharr_magnitude(&img));
    print_result("imageproc_gradients_mag", iterations, mag_elapsed);

    let pair_speedup = old_elapsed.as_secs_f64() / pair_elapsed.as_secs_f64();
    let mag_speedup = old_elapsed.as_secs_f64() / mag_elapsed.as_secs_f64();

    println!("ratio old/pair: {pair_speedup:.3}x");
    println!("ratio old/mag : {mag_speedup:.3}x (not equivalent work)");
}

fn main() {
    println!("Scharr benchmark: old manual loop vs imageproc");
    run_case(640, 480, 40);
    run_case(1280, 720, 20);
}
