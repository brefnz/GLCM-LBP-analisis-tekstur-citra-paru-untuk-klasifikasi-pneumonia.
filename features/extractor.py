"""
Feature Extraction Module
=========================
Combines three texture / statistical descriptors:
  1. GLCM  – Gray-Level Co-occurrence Matrix (6 properties × 2 distances × 4 angles)
  2. LBP   – Local Binary Pattern histogram (24-point uniform)
  3. STAT  – Pixel-level statistical moments

Total feature vector: ~96 dimensions
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis

from config import IMAGE_SIZE, GLCM_DISTANCES, GLCM_ANGLES_RAD, GLCM_LEVELS, LBP_RADIUS, LBP_N_POINTS


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale, resize, and apply CLAHE contrast enhancement."""
    if image is None:
        raise ValueError("Image is None – check the file path.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


# ─── GLCM ─────────────────────────────────────────────────────────────────────

def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Compute GLCM at 2 distances × 4 angles and extract 6 Haralick properties.
    Also appends mean and std across direction for each property.
    Returns a vector of length 6 * (2*4 + 2) = 6 * 10 = 60.
    """
    # Quantise to GLCM_LEVELS grey levels for speed
    quantised = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    np.clip(quantised, 0, GLCM_LEVELS - 1, out=quantised)

    glcm = graycomatrix(
        quantised,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES_RAD,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features: list[float] = []

    for prop in props:
        values = graycoprops(glcm, prop).flatten()   # shape: (n_dist × n_angles,)
        features.extend(values.tolist())
        features.append(float(np.mean(values)))
        features.append(float(np.std(values)))

    return np.array(features, dtype=np.float32)


# ─── LBP ──────────────────────────────────────────────────────────────────────

def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Compute uniform LBP and return a normalised histogram.
    Vector length: LBP_N_POINTS + 2 = 26.
    """
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    n_bins = LBP_N_POINTS + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )
    return hist.astype(np.float32)


# ─── Statistical ──────────────────────────────────────────────────────────────

def extract_statistical_features(gray: np.ndarray) -> np.ndarray:
    """
    First and higher-order pixel intensity statistics.
    Vector length: 10.
    """
    flat = gray.ravel().astype(np.float64)

    features = np.array([
        np.mean(flat),
        np.std(flat),
        float(skew(flat)),
        float(kurtosis(flat)),
        np.percentile(flat, 10),
        np.percentile(flat, 25),
        np.percentile(flat, 75),
        np.percentile(flat, 90),
        float(np.sum(flat ** 2)) / flat.size,   # normalised energy
        float(np.var(flat)),                     # variance
    ], dtype=np.float32)

    return features


# ─── Combined ─────────────────────────────────────────────────────────────────

def extract_features(source) -> np.ndarray:
    """
    Load (or accept) an image, preprocess, and concatenate all descriptors.

    Parameters
    ----------
    source : str | np.ndarray
        File path or BGR/grayscale image array.

    Returns
    -------
    np.ndarray, shape (n_features,)
    """
    if isinstance(source, str):
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f"Cannot read image: {source}")
    else:
        image = source

    gray = preprocess_image(image)

    return np.concatenate([
        extract_glcm_features(gray),   # 60
        extract_lbp_features(gray),    # 26
        extract_statistical_features(gray),  # 10
    ])
