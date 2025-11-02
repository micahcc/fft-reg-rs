use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use std::path::Path;

fn _log_polar_mapping(
    row: f32,
    col: f32,
    k_angle: f32,
    k_radius: f32,
    center: (f32, f32),
) -> (f32, f32) {
    // Inverse mapping function to convert from cartesian to polar coordinates
    let angle = col / k_angle;
    let rr = (row / k_radius).exp() * angle.sin() + center.0;
    let cc = (row / k_radius).exp() * angle.cos() + center.1;
    return (rr, cc);
}

fn warp_polar2(image: &GrayImage, radius: u32, output_shape: (u32, u32)) -> GrayImage {
    let (width, height) = output_shape;
    let center = (
        (image.height() as f32 / 2.0) - 0.5,
        (image.width() as f32 / 2.0) - 0.5,
    );
    let k_angle = height as f32 / (2.0 * PI);
    let k_radius = width as f32 / (radius as f32).ln();

    let mut warped_image = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let angle = y as f32 / k_angle;
            let rr = (x as f32 / k_radius).exp() * angle.sin() + center.0;
            let cc = (x as f32 / k_radius).exp() * angle.cos() + center.1;

            // Bilinear interpolation
            let x0 = cc.floor() as i32;
            let x1 = x0 + 1;
            let y0 = rr.floor() as i32;
            let y1 = y0 + 1;

            // Check if all interpolation points are within bounds
            if x0 >= 0 && (x1 as u32) < image.width() && y0 >= 0 && (y1 as u32) < image.height() {
                // Calculate interpolation weights
                let wx = cc - x0 as f32;
                let wy = rr - y0 as f32;

                // Perform bilinear interpolation
                let top = image.get_pixel(x0 as u32, y0 as u32).0[0] as f32 * (1.0 - wx)
                    + image.get_pixel(x1 as u32, y0 as u32).0[0] as f32 * wx;
                let bottom = image.get_pixel(x0 as u32, y1 as u32).0[0] as f32 * (1.0 - wx)
                    + image.get_pixel(x1 as u32, y1 as u32).0[0] as f32 * wx;
                let val = (top * (1.0 - wy) + bottom * wy) as u8;
                warped_image.put_pixel(x, y, Luma([val]));
            } else {
                // Handle boundary cases with nearest neighbor
                let input_x = cc.round() as i32;
                let input_y = rr.round() as i32;

                if input_x >= 0
                    && (input_x as u32) < image.width()
                    && input_y >= 0
                    && (input_y as u32) < image.height()
                {
                    warped_image.put_pixel(x, y, *image.get_pixel(input_x as u32, input_y as u32));
                } else {
                    warped_image.put_pixel(x, y, Luma([0]));
                }
            }
        }
    }

    warped_image
}

fn bandpass_1d(rel: f32) -> f32 {
    // 1d bandpass filter, after fourier shift
    (rel * 2.0 * PI).sin().powi(2)
}

fn bandpass_2d(width: u32, height: u32) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0; width as usize]; height as usize];
    for y in 0..height {
        for x in 0..width {
            let a = bandpass_1d(y as f32 / height as f32);
            let b = bandpass_1d(x as f32 / width as f32);
            out[y as usize][x as usize] = a + b;
        }
    }
    out
}

fn tukey_value(n: u32, size: u32, alpha: f32) -> f32 {
    let n = n as f32;
    let size = size as f32;

    if n < alpha * (size - 1.0) / 2.0 {
        0.5 * (1.0 + (PI * (2.0 * n / (alpha * (size - 1.0)) - 1.0)).cos())
    } else if n <= (size - 1.0) * (1.0 - alpha / 2.0) {
        1.0
    } else {
        0.5 * (1.0 + (PI * (2.0 * n / (alpha * (size - 1.0)) - 2.0 / alpha + 1.0)).cos())
    }
}

fn tukey_window_2d(width: u32, height: u32, alpha: f32) -> Vec<Vec<f32>> {
    let mut window_2d = vec![vec![0.0; width as usize]; height as usize];

    for i in 0..height {
        for j in 0..width {
            let tukey_i = tukey_value(i, height, alpha);
            let tukey_j = tukey_value(j, width, alpha);
            window_2d[i as usize][j as usize] = tukey_i * tukey_j;
        }
    }

    window_2d
}

// Helper function to convert GrayImage to a 2D vector of Complex<f32>
fn image_to_complex(img: &GrayImage) -> Vec<Vec<Complex<f32>>> {
    let (width, height) = (img.width(), img.height());
    let mut result = vec![vec![Complex::new(0.0, 0.0); width as usize]; height as usize];

    for y in 0..height {
        for x in 0..width {
            let val = img.get_pixel(x, y).0[0] as f32;
            result[y as usize][x as usize] = Complex::new(val, 0.0);
        }
    }

    result
}

// Helper function to apply a 2D window to a complex image
fn apply_window(img: &mut Vec<Vec<Complex<f32>>>, window: &Vec<Vec<f32>>) {
    for y in 0..img.len() {
        for x in 0..img[0].len() {
            img[y][x] *= window[y][x];
        }
    }
}

// Helper function to perform 2D FFT
fn fft2d(img: &mut Vec<Vec<Complex<f32>>>, inverse: bool) {
    let height = img.len();
    let width = img[0].len();
    let mut planner = FftPlanner::new();

    // First, perform FFT on each row
    let fft = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };

    for row in img.iter_mut() {
        fft.process(row);
        if inverse {
            // Scale inverse FFT
            for val in row.iter_mut() {
                *val /= width as f32;
            }
        }
    }

    // Transpose for column-wise FFT
    let mut transposed = vec![vec![Complex::new(0.0, 0.0); height]; width];
    for y in 0..height {
        for x in 0..width {
            transposed[x][y] = img[y][x];
        }
    }

    // Perform FFT on each column (now row in transposed matrix)
    let fft = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };

    for row in transposed.iter_mut() {
        fft.process(row);
        if inverse {
            // Scale inverse FFT
            for val in row.iter_mut() {
                *val /= height as f32;
            }
        }
    }

    // Transpose back
    for y in 0..height {
        for x in 0..width {
            img[y][x] = transposed[x][y];
        }
    }
}

// Helper function to perform FFT shift (swap quadrants)
fn fftshift(img: &mut Vec<Vec<Complex<f32>>>) {
    let height = img.len();
    let width = img[0].len();
    let half_height = height / 2;
    let half_width = width / 2;

    // Create a copy of the image
    let mut temp = vec![vec![Complex::new(0.0, 0.0); width]; height];
    for y in 0..height {
        for x in 0..width {
            temp[y][x] = img[y][x];
        }
    }

    // Swap quadrants
    // Upper-left with lower-right
    for y in 0..half_height {
        for x in 0..half_width {
            img[y][x] = temp[y + half_height][x + half_width];
            img[y + half_height][x + half_width] = temp[y][x];
        }
    }

    // Upper-right with lower-left
    for y in 0..half_height {
        for x in half_width..width {
            img[y][x] = temp[y + half_height][x - half_width];
            img[y + half_height][x - half_width] = temp[y][x];
        }
    }
}

// Helper function to get magnitude of complex image
fn complex_to_magnitude(img: &Vec<Vec<Complex<f32>>>) -> Vec<Vec<f32>> {
    let height = img.len();
    let width = img[0].len();
    let mut magnitude = vec![vec![0.0; width]; height];

    for y in 0..height {
        for x in 0..width {
            magnitude[y][x] = img[y][x].norm();
        }
    }

    magnitude
}

// Helper function to convert magnitude to GrayImage
fn magnitude_to_image(magnitude: &Vec<Vec<f32>>) -> GrayImage {
    let height = magnitude.len() as u32;
    let width = magnitude[0].len() as u32;
    let mut img = GrayImage::new(width, height);

    // Find max value for normalization
    let mut max_val: f32 = 0.0;
    for row in magnitude.iter() {
        for &val in row.iter() {
            max_val = max_val.max(val);
        }
    }

    // Normalize and convert to image
    for y in 0..height {
        for x in 0..width {
            let normalized = (magnitude[y as usize][x as usize] / max_val * 255.0).min(255.0);
            img.put_pixel(x, y, Luma([normalized as u8]));
        }
    }

    img
}

// Helper function for phase cross correlation
fn phase_cross_correlation(
    img1: &Vec<Vec<Complex<f32>>>,
    img2: &Vec<Vec<Complex<f32>>>,
) -> (f32, f32) {
    let height = img1.len();
    let width = img1[0].len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); width]; height];

    // Compute the product of img1 and conjugate of img2
    for y in 0..height {
        for x in 0..width {
            result[y][x] = img1[y][x] * img2[y][x].conj();
            // Normalize by magnitude
            let magnitude = result[y][x].norm();
            if magnitude > 1e-10 {
                result[y][x] /= magnitude;
            }
        }
    }

    // Inverse FFT
    fft2d(&mut result, true);

    // Find the maximum
    let mut max_val = 0.0;
    let mut max_y = 0;
    let mut max_x = 0;
    for y in 0..height {
        for x in 0..width {
            let val = result[y][x].norm();
            if val > max_val {
                max_val = val;
                max_y = y;
                max_x = x;
            }
        }
    }

    // Adjust for wrap-around
    let shift_y = if max_y > height / 2 {
        max_y as f32 - height as f32
    } else {
        max_y as f32
    };

    let shift_x = if max_x > width / 2 {
        max_x as f32 - width as f32
    } else {
        max_x as f32
    };

    (shift_y, shift_x)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters matching Python version
    let angle: f32 = 24.0;
    let scale = 1.4;
    let shiftr = 30;
    let shiftc = 15;

    // Load the image
    let img = image::open(Path::new("brick.jpg"))?.to_luma8();
    let (image_width, image_height) = (img.width(), img.height());

    // Create translated image
    let translated = imageproc::geometric_transformations::translate(&img, (shiftc, shiftr));

    // Create rotated image
    let rotated = imageproc::geometric_transformations::rotate_about_center(
        &translated,
        angle.to_radians(),
        Interpolation::Bilinear,
        Luma([0]),
    );

    // Rescale is more complex, we'll approximate it with a simple resize
    let rescaled = image::imageops::resize(
        &rotated,
        (rotated.width() as f32 * scale) as u32,
        (rotated.height() as f32 * scale) as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Crop to original size if larger
    let rts_image = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        if x < rescaled.width() && y < rescaled.height() {
            *rescaled.get_pixel(x, y)
        } else {
            Luma([0])
        }
    });

    // Create Tukey window
    let tukey_window = tukey_window_2d(image_width, image_height, 1.0);

    // Window images
    let mut wimage_complex = image_to_complex(&img);
    let mut rts_wimage_complex = image_to_complex(&rts_image);

    apply_window(&mut wimage_complex, &tukey_window);
    apply_window(&mut rts_wimage_complex, &tukey_window);

    // Perform FFT
    fft2d(&mut wimage_complex, false);
    fft2d(&mut rts_wimage_complex, false);

    // FFT shift
    fftshift(&mut wimage_complex);
    fftshift(&mut rts_wimage_complex);

    // Apply bandpass filter
    let bp_filt = bandpass_2d(image_width, image_height);
    for y in 0..wimage_complex.len() {
        for x in 0..wimage_complex[0].len() {
            wimage_complex[y][x] *= bp_filt[y][x];
            rts_wimage_complex[y][x] *= bp_filt[y][x];
        }
    }

    // Save filtered images (for visualization)
    let mut image_fs_filtered = wimage_complex.clone();
    let mut rts_fs_filtered = rts_wimage_complex.clone();

    fftshift(&mut image_fs_filtered);
    fftshift(&mut rts_fs_filtered);

    fft2d(&mut image_fs_filtered, true);
    fft2d(&mut rts_fs_filtered, true);

    // Get magnitudes for log-polar transform
    let image_fs_mag = complex_to_magnitude(&wimage_complex);
    let rts_fs_mag = complex_to_magnitude(&rts_wimage_complex);

    // Convert to gray images for warp_polar2
    let image_fs_gray = magnitude_to_image(&image_fs_mag);
    let rts_fs_gray = magnitude_to_image(&rts_fs_mag);

    // Create log-polar transformed FFT mag images
    let radius = image_height as u32 / 8; // only take lower frequencies
    let shape = (image_height as u32, image_width as u32);

    let warped_image_fs = warp_polar2(&image_fs_gray, radius, shape);
    let warped_rts_fs = warp_polar2(&rts_fs_gray, radius, shape);

    // Use only half of FFT
    let half_height = shape.0 / 2;
    let warped_image_fs_half = ImageBuffer::from_fn(shape.1, half_height, |x, y| {
        *warped_image_fs.get_pixel(x, y)
    });

    let warped_rts_fs_half =
        ImageBuffer::from_fn(shape.1, half_height, |x, y| *warped_rts_fs.get_pixel(x, y));

    // Convert back to complex for phase cross correlation
    let mut warped_image_complex = image_to_complex(&warped_image_fs_half);
    let mut warped_rts_complex = image_to_complex(&warped_rts_fs_half);

    // Perform FFT for phase cross correlation
    fft2d(&mut warped_image_complex, false);
    fft2d(&mut warped_rts_complex, false);

    // Calculate phase cross correlation
    let (shiftr, shiftc) = phase_cross_correlation(&warped_image_complex, &warped_rts_complex);

    // Calculate rotation and scaling
    let recovered_angle = (360.0 / shape.0 as f32) * shiftr;
    let klog = shape.1 as f32 / (radius as f32).ln();
    let shift_scale = (shiftc / klog).exp();

    // Save output images for visualization
    image_fs_gray.save("output_fft_original.png")?;
    rts_fs_gray.save("output_fft_transformed.png")?;

    let image_filtered_gray = magnitude_to_image(&complex_to_magnitude(&image_fs_filtered));
    let rts_filtered_gray = magnitude_to_image(&complex_to_magnitude(&rts_fs_filtered));

    image_filtered_gray.save("output_filtered_original.png")?;
    rts_filtered_gray.save("output_filtered_transformed.png")?;

    warped_image_fs.save("output_logpolar_original.png")?;
    warped_rts_fs.save("output_logpolar_transformed.png")?;

    warped_image_fs_half.save("output_logpolar_half_original.png")?;
    warped_rts_fs_half.save("output_logpolar_half_transformed.png")?;

    // Convert tukey window to image for visualization
    let tukey_window_img = ImageBuffer::from_fn(image_width, image_height, |x, y| {
        let val = (tukey_window[y as usize][x as usize] * 255.0) as u8;
        Luma([val])
    });
    tukey_window_img.save("output_tukey_window.png")?;

    // Create windowed images for visualization
    let windowed_original = ImageBuffer::from_fn(image_width, image_height, |x, y| {
        let val = (img.get_pixel(x, y).0[0] as f32 * tukey_window[y as usize][x as usize]) as u8;
        Luma([val])
    });

    let windowed_transformed = ImageBuffer::from_fn(image_width, image_height, |x, y| {
        let val =
            (rts_image.get_pixel(x, y).0[0] as f32 * tukey_window[y as usize][x as usize]) as u8;
        Luma([val])
    });

    windowed_original.save("output_windowed_original.png")?;
    windowed_transformed.save("output_windowed_transformed.png")?;

    // Save bandpass filter visualization
    let bandpass_img = ImageBuffer::from_fn(image_width, image_height, |x, y| {
        let val = (bp_filt[y as usize][x as usize] * 255.0) as u8;
        Luma([val])
    });
    bandpass_img.save("output_bandpass_filter.png")?;

    // Save original and transformed images
    img.save("output_original.png")?;
    rts_image.save("output_transformed.png")?;

    // Print results
    println!("Expected value for cc rotation in degrees: {}", angle);
    println!("Recovered value for cc rotation: {}", recovered_angle);
    println!();
    println!("Expected value for scaling difference: {}", scale);
    println!("Recovered value for scaling difference: {}", shift_scale);

    // Save results to a text file
    let result_text = format!(
        "Expected value for cc rotation in degrees: {}\n\
         Recovered value for cc rotation: {}\n\n\
         Expected value for scaling difference: {}\n\
         Recovered value for scaling difference: {}\n",
        angle, recovered_angle, scale, shift_scale
    );

    std::fs::write("output_results.txt", result_text)?;

    Ok(())
}
