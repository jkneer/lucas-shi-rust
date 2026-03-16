use image::{GrayImage, ImageBuffer, Luma};
use nalgebra::{DMatrix, DVector, SVD};

use crate::utils::fast_gradients::compute_gradients;

/// Compute optical flow using Lucas-Kanade method
///
/// # Arguments
/// * `prev_pyramid` - Previous frame (pyramid of grayscale)
/// * `curr_pyramid` - Next frame (pyramid of grayscale)
/// * `prev_points` - Feature points to track (in prev frame)
/// * `window_size` - Size of the search window (odd number)
/// * `max_iterations` - Max iterations for correct points on each layer
///
/// # Returns
/// Vector of points on next frame
pub fn calc_optical_flow(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert!(window_size % 2 == 1, "Window size must be odd");

    let n_levels = prev_pyramid.len();
    let radius = window_size / 2;
    let epsilon = 1e-3;

    // Initialize displacements to zero
    let mut displacements: Vec<(f32, f32)> = prev_points.iter().map(|_| (0.0, 0.0)).collect();

    // Process levels from top (coarse) to bottom (fine)
    for level in (0..n_levels).rev() {
        let scale = 2f32.powi(level as i32);

        // Get the images for the current level
        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];

        // Compute gradients for the previous image
        // let grad_x = horizontal_scharr(prev_img);
        // let grad_y = vertical_scharr(prev_img);
        // console_log!("{}", performance.now()-now);
        let (grad_x, grad_y) = compute_gradients(prev_img);

        // Process each point
        for ((prev_x, prev_y), disp) in prev_points.iter().zip(displacements.iter_mut()) {
            // Scale the original point for the current level
            let x = *prev_x / scale;
            let y = *prev_y / scale;

            // Add the current displacement, scaled for this level
            let mut dx = disp.0 / scale;
            let mut dy = disp.1 / scale;

            // Skip points outside image bounds
            if !in_bounds(prev_img, x, y, radius) {
                continue;
            }

            // Refine the displacement at the current level
            let mut converged = false;
            for _ in 0..max_iterations {
                if converged {
                    break;
                }

                // Compute the current position in the target image
                let curr_x = x + dx;
                let curr_y = y + dy;

                // Check bounds in the target image
                if !in_bounds(curr_img, curr_x, curr_y, radius) {
                    break;
                }

                // Collect data for the linear system
                let mut a_data = Vec::with_capacity(window_size * window_size * 2);
                let mut b_data = Vec::with_capacity(window_size * window_size);

                for j in -(radius as i32)..=radius as i32 {
                    for i in -(radius as i32)..=radius as i32 {
                        // Coordinates in the previous image
                        let px_prev = interpolate(prev_img, x + i as f32, y + j as f32);

                        // Coordinates in the current image with displacement applied
                        let px_curr = interpolate(curr_img, curr_x + i as f32, curr_y + j as f32);

                        // Gradients in the previous image (fixed)
                        let ix = interpolate_alt(&grad_x, x + i as f32, y + j as f32) / 32.0;
                        let iy = interpolate_alt(&grad_y, x + i as f32, y + j as f32) / 32.0;

                        a_data.push(ix);
                        a_data.push(iy);
                        b_data.push(px_prev - px_curr);
                    }
                }

                // Solve the linear system
                let n_pixels = window_size * window_size;
                if a_data.len() != 2 * n_pixels || b_data.len() != n_pixels {
                    break;
                }

                let a_matrix = DMatrix::from_row_slice(n_pixels, 2, &a_data);
                let b_vector = DVector::from_vec(b_data);

                let ata = a_matrix.transpose() * &a_matrix;
                let atb = a_matrix.transpose() * &b_vector;

                let svd = SVD::new(ata, true, true);
                if let Ok(solution) = svd.solve(&atb, 1e-6) {
                    let (ddx, ddy) = (solution[0], solution[1]);
                    dx += ddx;
                    dy += ddy;

                    if ddx.abs() < epsilon && ddy.abs() < epsilon {
                        converged = true;
                    }
                } else {
                    break;
                }
            }

            // Update the total displacement with the current level scale
            *disp = (dx * scale, dy * scale);
        }
    }

    // Return the final positions
    prev_points
        .iter()
        .zip(displacements.iter())
        .map(|((x, y), (dx, dy))| (x + dx, y + dy))
        .collect()
}

/// Checks that the window stays within image bounds
fn in_bounds(img: &GrayImage, x: f32, y: f32, radius: usize) -> bool {
    let (w, h) = (img.width() as f32, img.height() as f32);
    x >= radius as f32 && x < w - radius as f32 && y >= radius as f32 && y < h - radius as f32
}

/// Bilinear interpolation of the pixel value
fn interpolate(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);

        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };

        sum += px * wx * wy;
    }

    sum
}

fn interpolate_alt(img: &ImageBuffer<Luma<i16>, Vec<i16>>, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);

        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };

        sum += px * wx * wy;
    }

    sum
}
