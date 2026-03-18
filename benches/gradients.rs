#![cfg_attr(has_portable_simd, feature(portable_simd))]

// Main findings from this benchmark:
// - The historical Scharr loop is slowed down mostly by per-pixel `put_pixel` writes.
// - Writing into flat `Vec<i16>` buffers first and wrapping them with `ImageBuffer::from_vec`
//   avoids that overhead and was measurably faster on the x86_64 dev machine.
// - Explicit SIMD is the fastest path tested so far. On x86_64, AVX2 was roughly 5x to 6x
//   faster than the historical implementation in earlier runs.
// - On Raspberry Pi 5 (aarch64), the indexed scalar path was about 4.8x to 5.7x faster than
//   the historical implementation, and the NEON path was about 7.5x to 13.7x faster.
// - On nightly compilers, this bench also includes `std::simd` portable-SIMD rows:
//   a baseline row, an AVX2-targeted row, and an AVX-512-targeted row when the CPU supports them.
// - `imageproc`'s Scharr pair computes equivalent interior gradients, but border behavior differs:
//   the historical code leaves borders as zero while `imageproc` clamps out-of-bounds samples.
// - On Raspberry Pi 5, `imageproc_scharr_pair` and `gradients_grayscale` were both about 3x
//   slower than the historical implementation and much slower than the indexed/SIMD paths.
// - `gradients_grayscale` is not equivalent work because it computes gradient magnitude rather
//   than returning signed `Ix` and `Iy`.
//
// Treat timings as machine-dependent. Keep the correctness check enabled so all implementations
// continue to agree where they are supposed to.

use std::hint::black_box;
#[cfg(has_portable_simd)]
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use image::{GrayImage, ImageBuffer, Luma};
use imageproc::gradients::{gradients_grayscale, horizontal_scharr, vertical_scharr};
use imageproc::kernel::{SCHARR_HORIZONTAL_3X3, SCHARR_VERTICAL_3X3};
use optical_flow_lk::compute_gradients as crate_compute_gradients;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(has_portable_simd)]
use std::ptr;
#[cfg(has_portable_simd)]
use std::simd::num::SimdUint;
#[cfg(has_portable_simd)]
use std::simd::Simd;

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

fn manual_scharr_old_indexed(
    img: &GrayImage,
    kernel_x: &[i32; 9],
    kernel_y: &[i32; 9],
) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bottom = (y + 1) * width;

        for x in 1..width - 1 {
            let idx = mid + x;
            let gx = src[top + x - 1] as i32 * kernel_x[0]
                + src[top + x] as i32 * kernel_x[1]
                + src[top + x + 1] as i32 * kernel_x[2]
                + src[mid + x - 1] as i32 * kernel_x[3]
                + src[mid + x] as i32 * kernel_x[4]
                + src[mid + x + 1] as i32 * kernel_x[5]
                + src[bottom + x - 1] as i32 * kernel_x[6]
                + src[bottom + x] as i32 * kernel_x[7]
                + src[bottom + x + 1] as i32 * kernel_x[8];
            let gy = src[top + x - 1] as i32 * kernel_y[0]
                + src[top + x] as i32 * kernel_y[1]
                + src[top + x + 1] as i32 * kernel_y[2]
                + src[mid + x - 1] as i32 * kernel_y[3]
                + src[mid + x] as i32 * kernel_y[4]
                + src[mid + x + 1] as i32 * kernel_y[5]
                + src[bottom + x - 1] as i32 * kernel_y[6]
                + src[bottom + x] as i32 * kernel_y[7]
                + src[bottom + x + 1] as i32 * kernel_y[8];

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(target_arch = "aarch64")]
fn manual_scharr_old_simd(img: &GrayImage) -> GradientPair {
    unsafe { manual_scharr_old_simd_neon(img) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn manual_scharr_old_simd(img: &GrayImage) -> GradientPair {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            return manual_scharr_old_simd_avx2(img);
        }
    }

    manual_scharr_old_indexed(img, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
fn manual_scharr_old_simd(img: &GrayImage) -> GradientPair {
    manual_scharr_old_indexed(img, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD)
}

#[cfg(target_arch = "aarch64")]
fn simd_label() -> &'static str {
    "manual_scharr_old_neon"
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_label() -> &'static str {
    if is_x86_feature_detected!("avx2") {
        return "manual_scharr_old_simd";
    }

    "manual_scharr_old_simd_fallback"
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
fn simd_label() -> &'static str {
    "manual_scharr_old_simd_fallback"
}

#[cfg(has_portable_simd)]
fn manual_scharr_old_portable_simd(img: &GrayImage) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;

    if width < 3 || height < 3 {
        return (
            ImageBuffer::new(img.width(), img.height()),
            ImageBuffer::new(img.width(), img.height()),
        );
    }

    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = Simd::<i16, 16>::splat(3);
    let coeff10 = Simd::<i16, 16>::splat(10);
    let simd_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bottom = (y + 1) * width;
        let row = y * width;

        let mut x = 1usize;
        while x < simd_end {
            let tl = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x - 1)) };
            let tc = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x)) };
            let tr = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x + 1)) };
            let ml = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(mid + x - 1)) };
            let mr = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(mid + x + 1)) };
            let bl = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x - 1)) };
            let bc = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x)) };
            let br = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x + 1)) };

            let gx = ((tr + br) - (tl + bl)) * coeff3 + (mr - ml) * coeff10;
            let gy = ((bl + br) - (tl + tr)) * coeff3 + (bc - tc) * coeff10;

            unsafe {
                store_i16x16_simd(grad_x.as_mut_ptr().add(row + x), gx);
                store_i16x16_simd(grad_y.as_mut_ptr().add(row + x), gy);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(has_portable_simd)]
#[inline(always)]
unsafe fn load_u8x16_as_i16_simd(ptr_u8: *const u8) -> Simd<i16, 16> {
    let bytes = unsafe { ptr::read_unaligned(ptr_u8 as *const [u8; 16]) };
    Simd::<u8, 16>::from_array(bytes).cast::<i16>()
}

#[cfg(has_portable_simd)]
#[inline(always)]
unsafe fn store_i16x16_simd(ptr_i16: *mut i16, values: Simd<i16, 16>) {
    let values = values.to_array();
    unsafe { ptr::write_unaligned(ptr_i16 as *mut [i16; 16], values) };
}

#[cfg(has_portable_simd)]
#[inline(always)]
unsafe fn load_u8x32_as_i16_simd(ptr_u8: *const u8) -> Simd<i16, 32> {
    let bytes = unsafe { ptr::read_unaligned(ptr_u8 as *const [u8; 32]) };
    Simd::<u8, 32>::from_array(bytes).cast::<i16>()
}

#[cfg(has_portable_simd)]
#[inline(always)]
unsafe fn store_i16x32_simd(ptr_i16: *mut i16, values: Simd<i16, 32>) {
    let values = values.to_array();
    unsafe { ptr::write_unaligned(ptr_i16 as *mut [i16; 32], values) };
}

#[cfg(has_portable_simd)]
fn portable_simd_label() -> &'static str {
    "manual_scharr_old_portable"
}

#[cfg(has_portable_simd)]
fn portable_simd_enabled() -> bool {
    true
}

#[cfg(not(has_portable_simd))]
fn portable_simd_enabled() -> bool {
    false
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
fn portable_simd_avx2_enabled() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64"))))]
fn portable_simd_avx2_enabled() -> bool {
    false
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
fn portable_simd_avx512_enabled() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
}

#[cfg(not(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64"))))]
fn portable_simd_avx512_enabled() -> bool {
    false
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
fn portable_simd_avx2_label() -> &'static str {
    "manual_scharr_old_portable_avx2"
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
fn portable_simd_avx512_label() -> &'static str {
    "manual_scharr_old_portable_avx512"
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn manual_scharr_old_portable_simd_avx2(img: &GrayImage) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;

    if width < 3 || height < 3 {
        return (
            ImageBuffer::new(img.width(), img.height()),
            ImageBuffer::new(img.width(), img.height()),
        );
    }

    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = Simd::<i16, 16>::splat(3);
    let coeff10 = Simd::<i16, 16>::splat(10);
    let simd_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bottom = (y + 1) * width;
        let row = y * width;

        let mut x = 1usize;
        while x < simd_end {
            let tl = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x - 1)) };
            let tc = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x)) };
            let tr = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(top + x + 1)) };
            let ml = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(mid + x - 1)) };
            let mr = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(mid + x + 1)) };
            let bl = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x - 1)) };
            let bc = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x)) };
            let br = unsafe { load_u8x16_as_i16_simd(src.as_ptr().add(bottom + x + 1)) };

            let gx = ((tr + br) - (tl + bl)) * coeff3 + (mr - ml) * coeff10;
            let gy = ((bl + br) - (tl + tr)) * coeff3 + (bc - tc) * coeff10;

            unsafe {
                store_i16x16_simd(grad_x.as_mut_ptr().add(row + x), gx);
                store_i16x16_simd(grad_y.as_mut_ptr().add(row + x), gy);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn manual_scharr_old_portable_simd_avx512(img: &GrayImage) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;

    if width < 3 || height < 3 {
        return (
            ImageBuffer::new(img.width(), img.height()),
            ImageBuffer::new(img.width(), img.height()),
        );
    }

    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = Simd::<i16, 32>::splat(3);
    let coeff10 = Simd::<i16, 32>::splat(10);
    let simd_end = 1 + ((width - 2) / 32) * 32;

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bottom = (y + 1) * width;
        let row = y * width;

        let mut x = 1usize;
        while x < simd_end {
            let tl = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(top + x - 1)) };
            let tc = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(top + x)) };
            let tr = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(top + x + 1)) };
            let ml = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(mid + x - 1)) };
            let mr = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(mid + x + 1)) };
            let bl = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(bottom + x - 1)) };
            let bc = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(bottom + x)) };
            let br = unsafe { load_u8x32_as_i16_simd(src.as_ptr().add(bottom + x + 1)) };

            let gx = ((tr + br) - (tl + bl)) * coeff3 + (mr - ml) * coeff10;
            let gy = ((bl + br) - (tl + tr)) * coeff3 + (bc - tc) * coeff10;

            unsafe {
                store_i16x32_simd(grad_x.as_mut_ptr().add(row + x), gx);
                store_i16x32_simd(grad_y.as_mut_ptr().add(row + x), gy);
            }
            x += 32;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn manual_scharr_old_simd_avx2(img: &GrayImage) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = _mm256_set1_epi16(3);
    let coeff10 = _mm256_set1_epi16(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * width),
                src.as_ptr().add(y * width),
                src.as_ptr().add((y + 1) * width),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16(top.add(x - 1)),
                    load_u8x16_as_i16(top.add(x)),
                    load_u8x16_as_i16(top.add(x + 1)),
                    load_u8x16_as_i16(mid.add(x - 1)),
                    load_u8x16_as_i16(mid.add(x + 1)),
                    load_u8x16_as_i16(bottom.add(x - 1)),
                    load_u8x16_as_i16(bottom.add(x)),
                    load_u8x16_as_i16(bottom.add(x + 1)),
                )
            };

            let gx3 = _mm256_sub_epi16(_mm256_add_epi16(tr, br), _mm256_add_epi16(tl, bl));
            let gx10 = _mm256_sub_epi16(mr, ml);
            let gx = _mm256_add_epi16(
                _mm256_mullo_epi16(gx3, coeff3),
                _mm256_mullo_epi16(gx10, coeff10),
            );

            let gy3 = _mm256_sub_epi16(_mm256_add_epi16(bl, br), _mm256_add_epi16(tl, tr));
            let gy10 = _mm256_sub_epi16(bc, tc);
            let gy = _mm256_add_epi16(
                _mm256_mullo_epi16(gy3, coeff3),
                _mm256_mullo_epi16(gy10, coeff10),
            );

            unsafe {
                _mm256_storeu_si256(grad_x.as_mut_ptr().add(row + x) as *mut __m256i, gx);
                _mm256_storeu_si256(grad_y.as_mut_ptr().add(row + x) as *mut __m256i, gy);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn load_u8x16_as_i16(ptr: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(ptr as *const __m128i)) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn manual_scharr_old_simd_neon(img: &GrayImage) -> GradientPair {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let src = img.as_raw();
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = vdupq_n_s16(3);
    let coeff10 = vdupq_n_s16(10);
    let simd_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * width),
                src.as_ptr().add(y * width),
                src.as_ptr().add((y + 1) * width),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < simd_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16x8x2(top.add(x - 1)),
                    load_u8x16_as_i16x8x2(top.add(x)),
                    load_u8x16_as_i16x8x2(top.add(x + 1)),
                    load_u8x16_as_i16x8x2(mid.add(x - 1)),
                    load_u8x16_as_i16x8x2(mid.add(x + 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x - 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x)),
                    load_u8x16_as_i16x8x2(bottom.add(x + 1)),
                )
            };

            let gx_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.0, br.0), vaddq_s16(tl.0, bl.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.0, ml.0), coeff10),
            );
            let gx_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.1, br.1), vaddq_s16(tl.1, bl.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.1, ml.1), coeff10),
            );
            let gy_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.0, br.0), vaddq_s16(tl.0, tr.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.0, tc.0), coeff10),
            );
            let gy_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.1, br.1), vaddq_s16(tl.1, tr.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.1, tc.1), coeff10),
            );

            unsafe {
                vst1q_s16(grad_x.as_mut_ptr().add(row + x), gx_lo);
                vst1q_s16(grad_x.as_mut_ptr().add(row + x + 8), gx_hi);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x), gy_lo);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x + 8), gy_hi);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(img.width(), img.height(), grad_x).unwrap(),
        ImageBuffer::from_vec(img.width(), img.height(), grad_y).unwrap(),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load_u8x16_as_i16x8x2(ptr: *const u8) -> (int16x8_t, int16x8_t) {
    let bytes = unsafe { vld1q_u8(ptr) };
    (
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes))),
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bytes))),
    )
}

fn imageproc_scharr_pair(img: &GrayImage) -> GradientPair {
    (horizontal_scharr(img), vertical_scharr(img))
}

fn imageproc_scharr_magnitude(img: &GrayImage) -> ImageBuffer<Luma<u16>, Vec<u16>> {
    gradients_grayscale(img, SCHARR_HORIZONTAL_3X3, SCHARR_VERTICAL_3X3)
}

fn gradient_magnitude(dx: i16, dy: i16) -> u16 {
    ((dx as f32).powi(2) + (dy as f32).powi(2)).sqrt() as u16
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

fn assert_pairs_match_interior(expected: &GradientPair, actual: &GradientPair) {
    assert_eq!(expected.0.dimensions(), actual.0.dimensions());
    assert_eq!(expected.1.dimensions(), actual.1.dimensions());

    let (width, height) = expected.0.dimensions();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            assert_eq!(
                expected.0.get_pixel(x, y),
                actual.0.get_pixel(x, y),
                "horizontal mismatch at ({x}, {y})"
            );
            assert_eq!(
                expected.1.get_pixel(x, y),
                actual.1.get_pixel(x, y),
                "vertical mismatch at ({x}, {y})"
            );
        }
    }
}

fn assert_pairs_match_exact(expected: &GradientPair, actual: &GradientPair) {
    assert_eq!(expected.0, actual.0, "horizontal images differ");
    assert_eq!(expected.1, actual.1, "vertical images differ");
}

fn assert_magnitude_matches_pair(
    pair: &GradientPair,
    magnitude: &ImageBuffer<Luma<u16>, Vec<u16>>,
) {
    assert_eq!(pair.0.dimensions(), magnitude.dimensions());

    let (width, height) = magnitude.dimensions();
    for y in 0..height {
        for x in 0..width {
            let expected = Luma([gradient_magnitude(
                pair.0.get_pixel(x, y)[0],
                pair.1.get_pixel(x, y)[0],
            )]);
            assert_eq!(
                expected,
                *magnitude.get_pixel(x, y),
                "magnitude mismatch at ({x}, {y})"
            );
        }
    }
}

fn validate_implementations() {
    let img = make_test_image(128, 96);
    let manual = manual_scharr_old(&img, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD);
    let indexed =
        manual_scharr_old_indexed(&img, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD);
    let simd = manual_scharr_old_simd(&img);
    let live = crate_compute_gradients(&img);
    #[cfg(has_portable_simd)]
    let portable = manual_scharr_old_portable_simd(&img);
    let imageproc_pair = imageproc_scharr_pair(&img);
    let imageproc_mag = imageproc_scharr_magnitude(&img);

    assert_pairs_match_exact(&manual, &indexed);
    assert_pairs_match_exact(&manual, &simd);
    assert_pairs_match_exact(&manual, &live);
    #[cfg(has_portable_simd)]
    assert_pairs_match_exact(&manual, &portable);
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if portable_simd_avx2_enabled() {
        let portable_avx2 = unsafe { manual_scharr_old_portable_simd_avx2(&img) };
        assert_pairs_match_exact(&manual, &portable_avx2);
    }
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if portable_simd_avx512_enabled() {
        let portable_avx512 = unsafe { manual_scharr_old_portable_simd_avx512(&img) };
        assert_pairs_match_exact(&manual, &portable_avx512);
    }

    // Borders differ because the historical code leaves them at zero,
    // while imageproc clamps out-of-bounds samples.
    assert_pairs_match_interior(&manual, &imageproc_pair);
    assert_magnitude_matches_pair(&imageproc_pair, &imageproc_mag);
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

    let indexed_elapsed = time_it(iterations, || {
        manual_scharr_old_indexed(
            &img,
            black_box(&HORIZONTAL_SCHARR_3X3_OLD),
            black_box(&VERTICAL_SCHARR_3X3_OLD),
        )
    });
    print_result("manual_scharr_old_indexed", iterations, indexed_elapsed);

    let simd_elapsed = time_it(iterations, || manual_scharr_old_simd(&img));
    print_result(simd_label(), iterations, simd_elapsed);

    let live_elapsed = time_it(iterations, || crate_compute_gradients(&img));
    print_result("crate_compute_gradients", iterations, live_elapsed);

    #[cfg(has_portable_simd)]
    let portable_elapsed = time_it(iterations, || manual_scharr_old_portable_simd(&img));
    #[cfg(has_portable_simd)]
    print_result(portable_simd_label(), iterations, portable_elapsed);
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    let portable_avx2_elapsed = if portable_simd_avx2_enabled() {
        Some(time_it(iterations, || unsafe {
            manual_scharr_old_portable_simd_avx2(&img)
        }))
    } else {
        None
    };
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(elapsed) = portable_avx2_elapsed {
        print_result(portable_simd_avx2_label(), iterations, elapsed);
    }
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    let portable_avx512_elapsed = if portable_simd_avx512_enabled() {
        Some(time_it(iterations, || unsafe {
            manual_scharr_old_portable_simd_avx512(&img)
        }))
    } else {
        None
    };
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(elapsed) = portable_avx512_elapsed {
        print_result(portable_simd_avx512_label(), iterations, elapsed);
    }

    let pair_elapsed = time_it(iterations, || imageproc_scharr_pair(&img));
    print_result("imageproc_scharr_pair", iterations, pair_elapsed);

    let mag_elapsed = time_it(iterations, || imageproc_scharr_magnitude(&img));
    print_result("imageproc_gradients_mag", iterations, mag_elapsed);

    let pair_speedup = old_elapsed.as_secs_f64() / pair_elapsed.as_secs_f64();
    let mag_speedup = old_elapsed.as_secs_f64() / mag_elapsed.as_secs_f64();
    let indexed_speedup = old_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64();
    let simd_speedup = old_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64();
    let live_speedup = old_elapsed.as_secs_f64() / live_elapsed.as_secs_f64();

    println!("ratio old/indexed: {indexed_speedup:.3}x");
    println!("ratio old/simd  : {simd_speedup:.3}x");
    println!("ratio old/live  : {live_speedup:.3}x");
    #[cfg(has_portable_simd)]
    println!(
        "ratio old/portable: {:.3}x",
        old_elapsed.as_secs_f64() / portable_elapsed.as_secs_f64()
    );
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(elapsed) = portable_avx2_elapsed {
        println!(
            "ratio old/portable_avx2: {:.3}x",
            old_elapsed.as_secs_f64() / elapsed.as_secs_f64()
        );
    }
    #[cfg(all(has_portable_simd, any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(elapsed) = portable_avx512_elapsed {
        println!(
            "ratio old/portable_avx512: {:.3}x",
            old_elapsed.as_secs_f64() / elapsed.as_secs_f64()
        );
    }
    println!("ratio old/pair: {pair_speedup:.3}x");
    println!("ratio old/mag : {mag_speedup:.3}x (not equivalent work)");
}

fn main() {
    println!("Scharr benchmark: old manual loop vs imageproc");
    if portable_simd_enabled() {
        println!("portable std::simd bench enabled on this compiler");
    } else {
        println!("portable std::simd bench disabled on this compiler");
    }
    if portable_simd_avx2_enabled() {
        println!("portable std::simd AVX2 row enabled on this CPU");
    }
    if portable_simd_avx512_enabled() {
        println!("portable std::simd AVX-512 row enabled on this CPU");
    }
    validate_implementations();
    println!("Correctness check passed (interior equality; borders differ by design)");
    run_case(200, 150, 500);
    run_case(320, 240, 200);
    run_case(640, 480, 40);
    run_case(1280, 720, 20);
}
