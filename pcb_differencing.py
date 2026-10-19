"""Illumination-invariant and structural differencing for PCB inspection.

Provides:
- Raw pixel absolute difference
- Sobel gradient magnitude differencing (invariant to uniform brightness shifts)
- SSIM (Structural Similarity Index) differencing (illumination and contrast invariant)
- Hybrid fusion of Gradient and SSIM
- CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
- Illumination change simulation for robust testing
"""

import cv2
import numpy as np


def apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def compute_gradient_magnitude(img_gray, ksize=3):
    """Compute normalized Sobel gradient magnitude map."""
    float_img = img_gray.astype(np.float32)
    gx = cv2.Sobel(float_img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(float_img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def compute_ssim_difference(img1_gray, img2_gray, k1=0.01, k2=0.03, win_size=11, sigma=1.5, L=255.0):
    """Compute local Structural Similarity (SSIM) difference map.
    
    Identical regions have SSIM near 1.0 (difference near 0).
    Structural changes (defects) produce low SSIM (difference near 255).
    """
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    x = img1_gray.astype(np.float64)
    y = img2_gray.astype(np.float64)

    # Local means
    kernel_size = (win_size, win_size)
    mu_x = cv2.GaussianBlur(x, kernel_size, sigma)
    mu_y = cv2.GaussianBlur(y, kernel_size, sigma)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x_sq = cv2.GaussianBlur(x * x, kernel_size, sigma) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y * y, kernel_size, sigma) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, kernel_size, sigma) - mu_xy

    # Formula for SSIM
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)

    # Clamp SSIM to [-1, 1], invert to difference map [0, 255]
    ssim_map = np.clip(ssim_map, -1.0, 1.0)
    diff_map = ((1.0 - ssim_map) * 0.5 * 255.0).astype(np.uint8)
    return diff_map


def compute_difference_map(template_gray, test_gray, method="raw", use_clahe=False):
    """Compute difference map using selected illumination-invariant or classical method.
    
    Methods:
    - 'raw': Classical absolute difference |template - test|
    - 'gradient': Sobel gradient magnitude difference (resists uniform lighting shifts)
    - 'ssim': Structural similarity difference map (resists lighting and contrast shifts)
    - 'hybrid': Weighted combination of Gradient and SSIM differences
    """
    t_gray = template_gray.copy()
    s_gray = test_gray.copy()

    if use_clahe:
        t_gray = apply_clahe(t_gray)
        s_gray = apply_clahe(s_gray)

    method_lower = method.lower()
    if method_lower == "gradient":
        grad_t = compute_gradient_magnitude(t_gray)
        grad_s = compute_gradient_magnitude(s_gray)
        diff = cv2.absdiff(grad_t, grad_s)
    elif method_lower == "ssim":
        diff = compute_ssim_difference(t_gray, s_gray)
    elif method_lower == "hybrid":
        grad_t = compute_gradient_magnitude(t_gray)
        grad_s = compute_gradient_magnitude(s_gray)
        grad_diff = cv2.absdiff(grad_t, grad_s)
        ssim_diff = compute_ssim_difference(t_gray, s_gray)
        diff = cv2.addWeighted(grad_diff, 0.45, ssim_diff, 0.55, 0)
    else:
        # Default: raw absolute difference
        diff = cv2.absdiff(t_gray, s_gray)

    return diff


def simulate_illumination_change(image_bgr, brightness_offset=45, contrast_factor=1.25):
    """Artificially alter lighting to evaluate illumination invariance."""
    float_img = image_bgr.astype(np.float32)
    adjusted = float_img * contrast_factor + brightness_offset
    clipped = np.clip(adjusted, 0, 255).astype(np.uint8)
    return clipped
