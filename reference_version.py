from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

from skimage.filters import window, difference_of_gaussians
from scipy.fft import ifft2, fft2, fftshift
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, rescale


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

            input_x = int(round(cc))
            input_y = int(round(rr))

            if 0 <= input_x < image.shape[1] and 0 <= input_y < image.shape[0]:
                warped_image[y, x] = image[input_y, input_x]
            else:
                warped_image[y, x] = 0

    return warped_image


def bandpass_1d(rel):
    """
    1d bandpass filter, after fourier shift
    """
    # 0 1 0 1 0
    return math.sin(rel * 2 * math.pi) ** 2


def bandpass_2d(width, height):
    out = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            a = bandpass_1d(y / height)
            b = bandpass_1d(x / width)
            f = (a**2 + b**2) ** 0.5
            out[y, x] = f
    return out


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


def main():
    angle = 24
    scale = 1.4
    shiftr = 30
    shiftc = 15

    image = skimage.data.camera()
    # image = np.array(Image.open("brick.jpg"))

    translated = image[shiftr:, shiftc:]
    rotated = rotate(translated, angle)
    rescaled = rescale(rotated, scale)
    image_height, image_width = image.shape
    rts_image = rescaled[:image_height, :image_width]

    tukey_image = tukey_window_2d(image_width, image_height, 1.0)

    # window images
    wimage = image * tukey_image
    rts_wimage = rts_image * tukey_image

    # work with shifted FFT magnitudes
    image_fs = np.abs(fftshift(fft2(wimage)))
    rts_fs = np.abs(fftshift(fft2(rts_wimage)))

    bp_filt = bandpass_2d(image_width, image_height)

    image_fs = fftshift(fft2(wimage))
    rts_fs = fftshift(fft2(rts_wimage))

    image_fs = image_fs * bp_filt
    rts_fs = rts_fs * bp_filt

    # for vis
    image_filtered = np.abs(ifft2(fftshift(image_fs)))
    rts_filtered = np.abs(ifft2(fftshift(rts_fs)))
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

    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
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
    fig.suptitle("Working in frequency domain can recover rotation and scaling")
    plt.show()

    print(f"Expected value for cc rotation in degrees: {angle}")
    print(f"Recovered value for cc rotation: {recovered_angle}")
    print()
    print(f"Expected value for scaling difference: {scale}")
    print(f"Recovered value for scaling difference: {shift_scale}")


main()
