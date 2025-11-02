import math
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

from skimage.filters import window, difference_of_gaussians
from scipy.fft import ifft2, fft2, fftshift
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, rescale

ANGLE = 24
SCALE = 1.4
SHIFTR = 30
SHIFTC = 15


def laplacian_of_gaussians(width, height, sigma):
    out = np.zeros((height, width))
    scale = -1 / (math.pi * sigma**4)
    scale2 = 1 / (2 * sigma**2)
    for row in range(height):
        for col in range(width):
            x = col - width // 2
            y = row - height // 2

            mdist = (x**2 + y**2) / (2 * sigma**2)
            out[row, col] = scale * (1 - mdist) * math.exp(-mdist)
    return out


def make_log_filter(width, height, sigma):
    spatial = laplacian_of_gaussians(width, height, sigma)
    return fftshift(fft2(spatial))


def _log_polar_mapping(row, col, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates."""
    angle = col / k_angle
    rr = (np.exp(row / k_radius) * np.sin(angle)) + center[0]
    cc = (np.exp(row / k_radius) * np.cos(angle)) + center[1]
    return rr, cc


def warp_polar2(image, radius, output_shape):
    height, width = output_shape
    center = (np.array(image.shape)[:2] / 2) - 0.5
    k_angle = height / (2 * np.pi)

    k_radius = width / np.log(radius)
    warped_image = np.zeros(output_shape, dtype=image.dtype)

    for y in range(height):
        for x in range(width):
            angle = y / k_angle
            rr = (np.exp(x / k_radius)) * np.sin(angle) + center[0]
            cc = (np.exp(x / k_radius)) * np.cos(angle) + center[1]

            # Bilinear interpolation
            x0 = int(np.floor(cc))
            x1 = x0 + 1
            y0 = int(np.floor(rr))
            y1 = y0 + 1

            # Check if all interpolation points are within bounds
            if (
                0 <= x0 < image.shape[1]
                and 0 <= x1 < image.shape[1]
                and 0 <= y0 < image.shape[0]
                and 0 <= y1 < image.shape[0]
            ):

                # Calculate interpolation weights
                wx = cc - x0
                wy = rr - y0

                # Perform bilinear interpolation
                top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
                bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
                warped_image[y, x] = top * (1 - wy) + bottom * wy
            else:
                # Handle boundary cases with nearest neighbor
                input_x = int(round(cc))
                input_y = int(round(rr))

                if 0 <= input_x < image.shape[1] and 0 <= input_y < image.shape[0]:
                    warped_image[y, x] = image[input_y, input_x]
                else:
                    warped_image[y, x] = 0

    return warped_image


def tukey_value(n, size, alpha):
    """
    Compute the Tukey window value for a given index.

    Parameters:
    - n: Index for which to compute the Tukey window value.
    - size: Size of the window dimension.
    - alpha: Shape parameter of the Tukey window.

    Returns:
    - value: Tukey window value at index n.
    """
    if n < alpha * (size - 1) / 2:
        value = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (size - 1)) - 1)))
    elif n <= (size - 1) * (1 - alpha / 2):
        value = 1
    else:
        value = 0.5 * (
            1 + np.cos(np.pi * (2 * n / (alpha * (size - 1)) - 2 / alpha + 1))
        )
    return value


def tukey_window_2d(width, height, alpha):
    """
    Create a 2D Tukey window using a single nested for loop.

    Parameters:
    - alpha: Shape parameter of the Tukey window.

    Returns:
    - window_2d: 2D Tukey window.
    """
    window_2d = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            tukey_i = tukey_value(i, height, alpha)
            tukey_j = tukey_value(j, width, alpha)
            window_2d[i, j] = tukey_i * tukey_j

    return window_2d


def has_at_least_one_factor_of_three(n):
    return n % 3 == 0


def is_composed_of_2s_and_3s(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    return n == 1


def round_up_to_next_valid_number(n):
    while True:
        if is_composed_of_2s_and_3s(n) and has_at_least_one_factor_of_three(n):
            return n
        n += 1


def pad_for_fft(image):
    old_height, old_width = image.shape
    new_width = round_up_to_next_valid_number(image.shape[1])
    new_height = round_up_to_next_valid_number(image.shape[0])
    out = np.zeros((new_height, new_width))
    out[0:old_height, 0:old_width] = image[0:old_height, 0:old_width]
    return out


def apply_changes(image):
    # Apply some changes
    translated = image[SHIFTR:, SHIFTC:]
    rotated = rotate(translated, ANGLE)
    rescaled = rescale(rotated, SCALE)
    image_height, image_width = image.shape
    rts_image = rescaled[:image_height, :image_width]
    return rts_image


def main():
    image = np.array(Image.open("brick.jpg"))
    rts_image = apply_changes(image)

    image = pad_for_fft(image)
    rts_image = pad_for_fft(rts_image)
    image_height, image_width = image.shape

    tukey_image = tukey_window_2d(image_width, image_height, 1.0)
    bp_filt = make_log_filter(image_width, image_height, 5)

    # window images
    wimage = image * tukey_image
    rts_wimage = rts_image * tukey_image

    # work with shifted FFT magnitudes
    image_fs = np.abs(fftshift(fft2(wimage)))
    rts_fs = np.abs(fftshift(fft2(rts_wimage)))

    image_fs = fftshift(fft2(wimage))
    rts_fs = fftshift(fft2(rts_wimage))

    image_fs = image_fs * bp_filt
    rts_fs = rts_fs * bp_filt

    # for vis
    image_filtered = np.abs(fftshift(ifft2(fftshift(image_fs))))
    rts_filtered = np.abs(fftshift(ifft2(fftshift(rts_fs))))
    # for vis

    image_fs = np.abs(image_fs)
    rts_fs = np.abs(rts_fs)

    # Create log-polar transformed FFT mag images and register
    shape = image_fs.shape
    radius = shape[0] // 8  # only take lower frequencies
    # warped_image_fs = warp_polar(
    #    image_fs, radius=radius, output_shape=shape, scaling="log", order=0
    # )
    # warped_rts_fs = warp_polar(
    #    rts_fs, radius=radius, output_shape=shape, scaling="log", order=0
    # )
    warped_image_fs = warp_polar2(
        image_fs,
        radius=radius,
        output_shape=shape,
    )
    warped_rts_fs = warp_polar2(
        rts_fs,
        radius=radius,
        output_shape=shape,
    )

    warped_image_fs = warped_image_fs[: shape[0] // 2, :]  # only use half of FFT
    warped_rts_fs = warped_rts_fs[: shape[0] // 2, :]
    shifts, error, phasediff = phase_cross_correlation(
        warped_image_fs, warped_rts_fs, upsample_factor=10, normalization=None
    )

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    import ipdb

    ipdb.set_trace()
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    ax = axes.ravel()
    ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
    center = np.array(shape) // 2
    ax[0].imshow(
        image_fs[
            center[0] - radius : center[0] + radius,
            center[1] - radius : center[1] + radius,
        ],
        cmap="magma",
    )
    ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
    ax[1].imshow(
        rts_fs[
            center[0] - radius : center[0] + radius,
            center[1] - radius : center[1] + radius,
        ],
        cmap="magma",
    )
    ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
    ax[2].imshow(warped_image_fs, cmap="magma")
    ax[3].set_title("Log-Polar-Transformed\nModified FFT")
    ax[3].imshow(warped_rts_fs, cmap="magma")
    ax[4].set_title("Filtered 1")
    ax[4].imshow(image_filtered)
    ax[5].set_title("Filtered 2")
    ax[5].imshow(rts_filtered)
    ax[6].set_title("bandpass filter re")
    ax[6].imshow(bp_filt.imag)
    ax[7].set_title("bandpass filter re")
    ax[7].imshow(bp_filt.real)
    fig.suptitle("Working in frequency domain can recover rotation and scaling")
    plt.show()

    print(f"Expected value for cc rotation in degrees: {ANGLE}")
    print(f"Recovered value for cc rotation: {recovered_angle}")
    print()
    print(f"Expected value for scaling difference: {SCALE}")
    print(f"Recovered value for scaling difference: {shift_scale}")


main()
