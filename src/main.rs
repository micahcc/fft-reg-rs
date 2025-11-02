use image::{GrayImage, ImageBuffer, ImageFormat, Luma};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use std::path::Path;

// Global constants
const ANGLE: f32 = 24.0;
const SCALE: f32 = 1.4;
const SHIFTR: i32 = 30;
const SHIFTC: i32 = 15;
const SIGMA: f32 = 5.0; // sigma for LoG filter

#[derive(Clone, Debug)]
struct ComplexImage {
    width: usize,
    height: usize,
    pixels: Vec<Complex<f32>>,
}

impl ComplexImage {
    fn new(width: usize, height: usize) -> Self {
        ComplexImage {
            width,
            height,
            pixels: vec![Complex::new(0.0, 0.0); width * height],
        }
    }

    fn get_pixel(&self, x: usize, y: usize) -> &Complex<f32> {
        &self.pixels[y * self.width + x]
    }

    fn put_pixel(&mut self, x: usize, y: usize, v: Complex<f32>) {
        self.pixels[y * self.width + x] = v;
    }

    fn make_cropped(&self, xstart: usize, ystart: usize, width: usize, height: usize) -> Self {
        let mut out = Self::new(width, height);
        let xstop = (xstart + width).min(self.width);
        let ystop = (ystart + height).min(self.height);
        for y in ystart..ystop {
            for x in xstart..xstop {
                out.put_pixel(x, y, *self.get_pixel(x, y));
            }
        }
        out
    }

    // Helper function to save complex images as 16-bit PNGs with autoscaling
    fn save(&self, prefix: &str) {
        let height = self.height as u32;
        let width = self.width as u32;

        // Find min and max values for autoscaling
        let mut min_mag = f32::MAX;
        let mut max_mag = f32::MIN;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let complex = self.get_pixel(x, y);
                min_mag = min_mag.min(complex.norm());
                max_mag = max_mag.max(complex.norm());
            }
        }
        let range = (max_mag - min_mag) as f32;

        // Create 16-bit images
        let real_file = format!("{}_real.png", prefix);
        let imag_file = format!("{}_imag.png", prefix);
        let mag_file = format!("{}_mag.png", prefix);

        // Create 16-bit real part image
        let real_img = ImageBuffer::from_fn(width, height, |x, y| {
            let val = self.get_pixel(x as usize, y as usize).re;
            // Scale to 0-65535 range
            let scaled = ((val - min_mag) / range * 65535.0) as u16;
            Luma([scaled])
        });

        // Create 16-bit imaginary part image
        let imag_img = ImageBuffer::from_fn(width, height, |x, y| {
            let val = self.get_pixel(x as usize, y as usize).im;
            // Scale to 0-65535 range
            let scaled = ((val - min_mag) / range * 65535.0) as u16;
            Luma([scaled])
        });

        // Create 16-bit imaginary part image
        let mag_img = ImageBuffer::from_fn(width, height, |x, y| {
            let val = self.get_pixel(x as usize, y as usize).norm();
            // Scale to 0-65535 range
            let scaled = ((val - min_mag) / range * 65535.0) as u16;
            Luma([scaled])
        });

        // Save images
        real_img
            .save_with_format(&real_file, ImageFormat::Png)
            .unwrap();
        imag_img
            .save_with_format(&imag_file, ImageFormat::Png)
            .unwrap();
        mag_img
            .save_with_format(&mag_file, ImageFormat::Png)
            .unwrap();
    }
}

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

fn warp_polar2(image: &ComplexImage, radius: u32, width: usize, height: usize) -> ComplexImage {
    let center = (
        (image.height as f32 / 2.0) - 0.5,
        (image.width as f32 / 2.0) - 0.5,
    );
    let k_angle = height as f32 / (2.0 * PI);
    let k_radius = width as f32 / (radius as f32).ln();

    let mut warped_image = ComplexImage::new(width, height);

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
            if x0 >= 0 && (x1 as usize) < image.width && y0 >= 0 && (y1 as usize) < image.height {
                // Calculate interpolation weights
                let wx = cc - x0 as f32;
                let wy = rr - y0 as f32;

                // Perform bilinear interpolation
                let top = image.get_pixel(x0 as usize, y0 as usize) * (1.0 - wx)
                    + image.get_pixel(x1 as usize, y0 as usize) * wx;
                let bottom = image.get_pixel(x0 as usize, y1 as usize) * (1.0 - wx)
                    + image.get_pixel(x1 as usize, y1 as usize) * wx;
                let val = top * (1.0 - wy) + bottom * wy;
                warped_image.put_pixel(x, y, val);
            } else {
                // Handle boundary cases with nearest neighbor
                let input_x = cc.round() as i32;
                let input_y = rr.round() as i32;

                if input_x >= 0
                    && (input_x as usize) < image.width
                    && input_y >= 0
                    && (input_y as usize) < image.height
                {
                    warped_image.put_pixel(
                        x,
                        y,
                        *image.get_pixel(input_x as usize, input_y as usize),
                    );
                } else {
                    warped_image.put_pixel(x, y, Complex::new(0.0, 0.0));
                }
            }
        }
    }

    warped_image
}

fn laplacian_of_gaussians(width: usize, height: usize, sigma: f32) -> ComplexImage {
    let mut out = ComplexImage::new(width, height);
    let scale = -1.0 / (PI * sigma.powi(4));
    let scale2 = 1.0 / (2.0 * sigma * sigma);

    for row in 0..height {
        for col in 0..width {
            let x = col as i32 - (width as i32 / 2);
            let y = row as i32 - (height as i32 / 2);

            let mdist = (x.pow(2) + y.pow(2)) as f32 * scale2;
            out.put_pixel(
                col,
                row,
                Complex::new(scale * (1.0 - mdist) * (-mdist).exp(), 0.0),
            );
        }
    }

    out
}

fn make_log_filter(width: usize, height: usize, sigma: f32) -> ComplexImage {
    let mut complex_spatial = laplacian_of_gaussians(width, height, sigma);

    fft2d(&mut complex_spatial);
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

// Helper function to convert GrayImage to ComplexImage
fn image_to_complex(img: &GrayImage) -> ComplexImage {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let mut result = ComplexImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let val = img.get_pixel(x as u32, y as u32).0[0] as f32;
            result.put_pixel(x, y, Complex::new(val, 0.0));
        }
    }

    result
}

// Helper function to apply a 2D window to a complex image
fn apply_window(img: &mut ComplexImage, window: &Vec<Vec<f32>>) {
    for y in 0..img.height {
        for x in 0..img.width {
            let pixel = img.get_pixel(x, y);
            img.put_pixel(x, y, *pixel * window[y][x]);
        }
    }
}

fn ifft2d(img: &mut ComplexImage) {
    let height = img.height;
    let width = img.width;
    let mut planner = FftPlanner::new();

    // First, perform FFT on each row
    let fft = planner.plan_fft_inverse(width);

    // Process rows
    let mut tmp = vec![Complex::new(0.0, 0.0); width];
    for y in 0..height {
        for x in 0..width {
            tmp[x] = *img.get_pixel(x, y);
        }
        fft.process(&mut tmp);
        for (x, val) in tmp.iter_mut().enumerate() {
            img.put_pixel(x, y, *val);
        }
    }

    // Perform FFT on each column (now row in transposed matrix)
    let fft = planner.plan_fft_inverse(height);

    // Process columns (rows in transposed image)
    tmp.resize(height, Complex::new(0.0, 0.0));
    for x in 0..width {
        for y in 0..height {
            tmp[y] = *img.get_pixel(x, y);
        }
        fft.process(&mut tmp);
        for (y, val) in tmp.iter_mut().enumerate() {
            img.put_pixel(x, y, *val);
        }
    }
}

fn fft2d(img: &mut ComplexImage) {
    let height = img.height;
    let width = img.width;
    let mut planner = FftPlanner::new();

    // First, perform FFT on each row
    let fft = planner.plan_fft_forward(width);

    // Process rows
    let mut tmp = vec![Complex::new(0.0, 0.0); width];
    for y in 0..height {
        for x in 0..width {
            tmp[x] = *img.get_pixel(x, y);
        }
        fft.process(&mut tmp);
        for (x, val) in tmp.iter_mut().enumerate() {
            img.put_pixel(x, y, *val);
        }
    }

    // Perform FFT on each column (now row in transposed matrix)
    let fft = planner.plan_fft_forward(height);

    // Process columns (rows in transposed image)
    tmp.resize(height, Complex::new(0.0, 0.0));
    for x in 0..width {
        for y in 0..height {
            tmp[y] = *img.get_pixel(x, y);
        }
        fft.process(&mut tmp);
        for (y, val) in tmp.iter_mut().enumerate() {
            img.put_pixel(x, y, *val);
        }
    }
}

// Helper function to perform FFT shift (swap quadrants)
fn fftshift(img: &mut ComplexImage) {
    let height = img.height;
    let width = img.width;
    let half_height = height / 2;
    let half_width = width / 2;

    // Create a copy of the image
    let mut temp = ComplexImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            temp.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }

    // Swap quadrants
    // Upper-left with lower-right
    for y in 0..half_height {
        for x in 0..half_width {
            img.put_pixel(x, y, *temp.get_pixel(x + half_width, y + half_height));
            img.put_pixel(x + half_width, y + half_height, *temp.get_pixel(x, y));
        }
    }

    // Upper-right with lower-left
    for y in 0..half_height {
        for x in half_width..width {
            img.put_pixel(x, y, *temp.get_pixel(x - half_width, y + half_height));
            img.put_pixel(x - half_width, y + half_height, *temp.get_pixel(x, y));
        }
    }
}

// Helper function to get magnitude of complex image
fn complex_to_magnitude(img: &ComplexImage) -> ComplexImage {
    let height = img.height;
    let width = img.width;
    let mut magnitude = ComplexImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            magnitude.put_pixel(x, y, Complex::new(img.get_pixel(x, y).norm(), 0.0));
        }
    }

    magnitude
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
    let rotated = imageproc::geometric_transformations::rotate_about_center(
        &translated,
        ANGLE.to_radians(),
        imageproc::geometric_transformations::Interpolation::Bilinear,
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

fn phase_cross_correlation(img1: &ComplexImage, img2: &ComplexImage, name: &str) -> (f32, f32) {
    let height = img1.height;
    let width = img1.width;
    let mut result = ComplexImage::new(width, height);

    img1.save(&format!("img1_{name}"));
    img2.save(&format!("img2_{name}"));

    // Perform FFT for phase cross correlation
    let mut img1_f = img1.clone();
    let mut img2_f = img2.clone();
    fft2d(&mut img1_f);
    fft2d(&mut img2_f);

    img1_f.save(&format!("img1_f_{name}"));
    img2_f.save(&format!("img2_f_{name}"));

    // Compute the product of img1 and conjugate of img2
    for y in 0..height {
        for x in 0..width {
            // Normalize by magnitude
            let val = img1_f.get_pixel(x, y) * img2_f.get_pixel(x, y).conj();
            let val = val / val.norm();
            result.put_pixel(x, y, val);
        }
    }

    ifft2d(&mut result);

    // Save cross-correlation FFT data
    result.save(&format!("corr_{name}"));

    // Find the maximum
    let mut max_val = 0.0;
    let mut max_y = 0;
    let mut max_x = 0;
    for y in 0..height {
        for x in 0..width {
            let val = result.get_pixel(x, y).re;
            if val > max_val {
                max_val = val;
                max_y = y;
                max_x = x;
            }
        }
    }

    eprintln!("raw max: {max_x} {max_y} {max_val}");

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
    let bp_filt = make_log_filter(image_width as usize, image_height as usize, SIGMA);

    // Window images
    let mut wimage_complex = image_to_complex(&img_padded);
    let mut rts_wimage_complex = image_to_complex(&rts_image_padded);

    apply_window(&mut wimage_complex, &tukey_window);
    apply_window(&mut rts_wimage_complex, &tukey_window);
    wimage_complex.save("target_windowed");
    rts_wimage_complex.save("mod_windowed");

    // Perform FFT
    fft2d(&mut wimage_complex);
    fft2d(&mut rts_wimage_complex);

    // FFT shift
    fftshift(&mut wimage_complex);
    fftshift(&mut rts_wimage_complex);

    // Save original FFT data
    wimage_complex.save("fft_orig");
    rts_wimage_complex.save("fft_transformed");

    // Apply Laplacian of Gaussians bandpass filter
    bp_filt.save("bandpass");
    for y in 0..image_height {
        for x in 0..image_width {
            let x = x as usize;
            let y = y as usize;
            let f = bp_filt.get_pixel(x, y);
            wimage_complex.put_pixel(x, y, wimage_complex.get_pixel(x, y) * f);
            rts_wimage_complex.put_pixel(x, y, rts_wimage_complex.get_pixel(x, y) * f);
        }
    }
    // Save filtered FFT data
    wimage_complex.save("fft_filtered_orig");
    rts_wimage_complex.save("fft_filtered_transformed");

    {
        // Save filtered images (for visualization)
        let mut image_fs_filtered = wimage_complex.clone();
        let mut rts_fs_filtered = rts_wimage_complex.clone();

        fftshift(&mut image_fs_filtered);
        fftshift(&mut rts_fs_filtered);

        ifft2d(&mut image_fs_filtered);
        ifft2d(&mut rts_fs_filtered);

        fftshift(&mut image_fs_filtered);
        fftshift(&mut rts_fs_filtered);

        // Save data after inverse FFT
        image_fs_filtered.save("ifft_orig");
        rts_fs_filtered.save("ifft_transformed");
    }

    // Get magnitudes for log-polar transform
    let image_fs_mag = complex_to_magnitude(&wimage_complex);
    let rts_fs_mag = complex_to_magnitude(&rts_wimage_complex);

    // Create log-polar transformed FFT mag images
    let radius = image_height as u32 / 8; // only take lower frequencies

    let mut warped_image_fs = warp_polar2(
        &image_fs_mag,
        radius,
        image_width as usize,
        image_height as usize,
    );
    let mut warped_rts_fs = warp_polar2(
        &rts_fs_mag,
        radius,
        image_width as usize,
        image_height as usize,
    );

    // Use only half of FFT

    // Calculate phase cross correlation
    apply_window(&mut warped_image_fs, &tukey_window);
    apply_window(&mut warped_rts_fs, &tukey_window);
    let (shiftr, shiftc) = phase_cross_correlation(&warped_image_fs, &warped_rts_fs, "warp");

    // Calculate rotation and scaling
    let recovered_angle = (360.0 / image_height as f32) * shiftr;
    let klog = image_width as f32 / (radius as f32).ln();
    let shift_scale = (shiftc / klog).exp();
    eprintln!(
        "Log polar shift: {shiftr}, {shiftc}, angle: {recovered_angle}, scale: {shift_scale}"
    );

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
