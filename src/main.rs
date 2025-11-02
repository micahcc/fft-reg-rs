use image::{GenericImageView, GrayImage, ImageBuffer, ImageFormat, Luma};
use imageproc::geometric_transformations::{rotate_about_center, translate, Interpolation};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use std::fs;
use std::io;
use std::path::Path;

// Global constants
const ANGLE: f32 = 24.0;
const SCALE: f32 = 1.4;
const SHIFTR: i32 = 30;
const SHIFTC: i32 = 15;
const SIGMA: f32 = 5.0; // sigma for LoG filter

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

fn laplacian_of_gaussians(width: u32, height: u32, sigma: f32) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0; width as usize]; height as usize];
    let scale = -1.0 / (PI * sigma.powi(4));
    let scale2 = 1.0 / (2.0 * sigma * sigma);

    for row in 0..height {
        for col in 0..width {
            let x = col as i32 - (width as i32 / 2);
            let y = row as i32 - (height as i32 / 2);

            let mdist = (x.pow(2) + y.pow(2)) as f32 * scale2;
            out[row as usize][col as usize] = scale * (1.0 - mdist) * (-mdist).exp();
        }
    }

    out
}

fn make_log_filter(width: u32, height: u32, sigma: f32) -> Vec<Vec<Complex<f32>>> {
    let spatial = laplacian_of_gaussians(width, height, sigma);

    // Convert to complex
    let mut complex_spatial = vec![vec![Complex::new(0.0, 0.0); width as usize]; height as usize];
    for y in 0..height {
        for x in 0..width {
            complex_spatial[y as usize][x as usize] =
                Complex::new(spatial[y as usize][x as usize], 0.0);
        }
    }

    // Perform FFT
    fft2d(&mut complex_spatial, false);
    fftshift(&mut complex_spatial);

    complex_spatial
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

// Helper function to save complex images as 16-bit PNGs with autoscaling
fn save_complex_image(complex_data: &Vec<Vec<Complex<f32>>>, prefix: &str) -> io::Result<()> {
    let height = complex_data.len() as u32;
    let width = complex_data[0].len() as u32;

    // Extract real and imaginary parts
    let mut real_values = vec![vec![0.0; width as usize]; height as usize];
    let mut imag_values = vec![vec![0.0; width as usize]; height as usize];

    // Find min and max values for autoscaling
    let mut min_real = f32::MAX;
    let mut max_real = f32::MIN;
    let mut min_imag = f32::MAX;
    let mut max_imag = f32::MIN;

    for y in 0..height {
        for x in 0..width {
            let complex = complex_data[y as usize][x as usize];
            let real = complex.re;
            let imag = complex.im;

            real_values[y as usize][x as usize] = real;
            imag_values[y as usize][x as usize] = imag;

            min_real = min_real.min(real);
            max_real = max_real.max(real);
            min_imag = min_imag.min(imag);
            max_imag = max_imag.max(imag);
        }
    }

    // Create 16-bit images
    let real_file = format!("{}_real.png", prefix);
    let imag_file = format!("{}_imag.png", prefix);

    // Create 16-bit real part image
    let real_img = ImageBuffer::from_fn(width, height, |x, y| {
        let val = real_values[y as usize][x as usize];
        // Scale to 0-65535 range
        let range = max_real - min_real;
        let scaled = if range > 1e-10 {
            ((val - min_real) / range * 65535.0) as u16
        } else {
            // Handle flat data
            32768 as u16
        };
        Luma([scaled])
    });

    // Create 16-bit imaginary part image
    let imag_img = ImageBuffer::from_fn(width, height, |x, y| {
        let val = imag_values[y as usize][x as usize];
        // Scale to 0-65535 range
        let range = max_imag - min_imag;
        let scaled = if range > 1e-10 {
            ((val - min_imag) / range * 65535.0) as u16
        } else {
            // Handle flat data
            32768 as u16
        };
        Luma([scaled])
    });

    // Save images
    real_img
        .save_with_format(&real_file, ImageFormat::Png)
        .unwrap();
    imag_img
        .save_with_format(&imag_file, ImageFormat::Png)
        .unwrap();

    Ok(())
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
fn has_at_least_one_factor_of_three(n: usize) -> bool {
    n % 3 == 0
}

fn is_composed_of_2s_and_3s(n: usize) -> bool {
    let mut num = n;
    while num % 2 == 0 {
        num /= 2;
    }
    while num % 3 == 0 {
        num /= 3;
    }
    num == 1
}

fn round_up_to_next_valid_number(n: usize) -> usize {
    let mut num = n;
    while !(is_composed_of_2s_and_3s(num) && has_at_least_one_factor_of_three(num)) {
        num += 1;
    }
    num
}

fn pad_for_fft(image: &GrayImage) -> GrayImage {
    let old_width = image.width() as usize;
    let old_height = image.height() as usize;
    let new_width = round_up_to_next_valid_number(old_width);
    let new_height = round_up_to_next_valid_number(old_height);

    let mut padded = ImageBuffer::new(new_width as u32, new_height as u32);

    // Copy original image data
    for y in 0..old_height {
        for x in 0..old_width {
            padded.put_pixel(x as u32, y as u32, *image.get_pixel(x as u32, y as u32));
        }
    }

    padded
}

fn apply_changes(image: &GrayImage) -> GrayImage {
    // Apply transformations: translate, rotate, scale
    let translated = imageproc::geometric_transformations::translate(image, (SHIFTC, SHIFTR));

    // Create rotated image
    let rotated = rotate_about_center(
        &translated,
        ANGLE.to_radians(),
        Interpolation::Bilinear,
        Luma([0]),
    );

    // Rescale is more complex, we'll approximate it with a simple resize
    let rescaled = image::imageops::resize(
        &rotated,
        (rotated.width() as f32 * SCALE) as u32,
        (rotated.height() as f32 * SCALE) as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Crop to original size if larger
    let result = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        if x < rescaled.width() && y < rescaled.height() {
            *rescaled.get_pixel(x, y)
        } else {
            Luma([0])
        }
    });

    result
}

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
    // Load the image
    let img = image::open(Path::new("brick.jpg"))?.to_luma8();

    // Apply transformations
    let rts_image = apply_changes(&img);

    // Pad both images for FFT
    let img_padded = pad_for_fft(&img);
    let rts_image_padded = pad_for_fft(&rts_image);
    let (image_width, image_height) = (img_padded.width(), img_padded.height());

    // Create Tukey window
    let tukey_window = tukey_window_2d(image_width, image_height, 1.0);

    // Window images
    let mut wimage_complex = image_to_complex(&img_padded);
    let mut rts_wimage_complex = image_to_complex(&rts_image_padded);

    apply_window(&mut wimage_complex, &tukey_window);
    apply_window(&mut rts_wimage_complex, &tukey_window);

    // Perform FFT
    fft2d(&mut wimage_complex, false);
    fft2d(&mut rts_wimage_complex, false);

    // FFT shift
    fftshift(&mut wimage_complex);
    fftshift(&mut rts_wimage_complex);

    // Save original FFT data
    save_complex_image(&wimage_complex, "fft_orig")?;
    save_complex_image(&rts_wimage_complex, "fft_transformed")?;

    // Apply Laplacian of Gaussians bandpass filter
    let bp_filt = make_log_filter(image_width, image_height, SIGMA);
    save_complex_image(&bp_filt, "bandpass")?;
    for y in 0..wimage_complex.len() {
        for x in 0..wimage_complex[0].len() {
            wimage_complex[y][x] *= bp_filt[y][x];
            rts_wimage_complex[y][x] *= bp_filt[y][x];
        }

        // Save filtered FFT data
        save_complex_image(&wimage_complex, "fft_filtered_orig")?;
        save_complex_image(&rts_wimage_complex, "fft_filtered_transformed")?;
    }

    // Save filtered images (for visualization)
    let mut image_fs_filtered = wimage_complex.clone();
    let mut rts_fs_filtered = rts_wimage_complex.clone();

    fftshift(&mut image_fs_filtered);
    fftshift(&mut rts_fs_filtered);

    // Save shifted data before inverse FFT
    save_complex_image(&image_fs_filtered, "fft_shifted_orig")?;
    save_complex_image(&rts_fs_filtered, "fft_shifted_transformed")?;

    fft2d(&mut image_fs_filtered, true);
    fft2d(&mut rts_fs_filtered, true);

    // Save data after inverse FFT
    save_complex_image(&image_fs_filtered, "ifft_orig")?;
    save_complex_image(&rts_fs_filtered, "ifft_transformed")?;

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

    // Save cross-correlation FFT data
    save_complex_image(&warped_image_complex, "cross_corr_orig")?;
    save_complex_image(&warped_rts_complex, "cross_corr_transformed")?;

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

    // Save original and transformed images
    img.save("output_original.png")?;
    rts_image.save("output_transformed.png")?;

    // Print results
    println!("Expected value for cc rotation in degrees: {}", ANGLE);
    println!("Recovered value for cc rotation: {}", recovered_angle);
    println!();
    println!("Expected value for scaling difference: {}", SCALE);
    println!("Recovered value for scaling difference: {}", shift_scale);

    // Save results to a text file
    let result_text = format!(
        "Expected value for cc rotation in degrees: {}\n\
         Recovered value for cc rotation: {}\n\n\
         Expected value for scaling difference: {}\n\
         Recovered value for scaling difference: {}\n",
        ANGLE, recovered_angle, SCALE, shift_scale
    );

    std::fs::write("output_results.txt", result_text)?;

    Ok(())
}
