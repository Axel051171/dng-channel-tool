"""
Weißabgleich-Pipette (#2) und Kamera-JPEG-Vergleich (#8)

- Klick auf neutralen Punkt → WB-Korrektur berechnen
- Eingebettetes JPEG aus RAW extrahieren und vergleichen
"""

import numpy as np
from typing import Tuple, Optional


def calculate_wb_from_pixel(image: np.ndarray, x: int, y: int,
                              sample_size: int = 10) -> dict:
    """
    Berechnet Weißabgleich-Korrektur aus einem angeklickten Punkt.

    Der angeklickte Punkt sollte neutral-grau sein.
    Die Korrektur macht diesen Punkt zu reinem Grau.

    Args:
        image: RGB numpy array
        x, y: Klick-Koordinaten
        sample_size: Messfeldgröße

    Returns:
        dict mit: r_gain, g_gain, b_gain, temperature, tint,
                  lr_temperature, lr_tint, nikon_rb_coeff
    """
    h, w = image.shape[:2]
    half = sample_size // 2

    y1 = max(0, y - half)
    y2 = min(h, y + half)
    x1 = max(0, x - half)
    x2 = min(w, x + half)

    patch = image[y1:y2, x1:x2].astype(np.float64)
    avg_r = np.mean(patch[:, :, 0])
    avg_g = np.mean(patch[:, :, 1])
    avg_b = np.mean(patch[:, :, 2])

    # Avoid division by zero
    avg_r = max(avg_r, 1)
    avg_g = max(avg_g, 1)
    avg_b = max(avg_b, 1)

    # Gains to make this pixel neutral gray
    # Normalize to green channel (most common reference)
    target = avg_g  # Green as reference
    r_gain = target / avg_r
    g_gain = 1.0
    b_gain = target / avg_b

    # Estimate color temperature from R/B ratio
    rb_ratio = avg_r / avg_b
    # Approximate: high R/B = warm (low K), low R/B = cool (high K)
    # Very rough mapping based on blackbody radiation
    if rb_ratio > 1.5:
        est_temp = 3000  # Very warm
    elif rb_ratio > 1.2:
        est_temp = 4000  # Warm
    elif rb_ratio > 1.0:
        est_temp = 5000  # Slightly warm
    elif rb_ratio > 0.8:
        est_temp = 6500  # Neutral
    elif rb_ratio > 0.6:
        est_temp = 8000  # Cool
    else:
        est_temp = 10000  # Very cool

    # Correction temperature (inverse of what we measured)
    if rb_ratio > 1.0:
        correction_temp = int(6500 + (1.0 - rb_ratio) * 5000)
    else:
        correction_temp = int(6500 + (1.0 - rb_ratio) * 5000)

    correction_temp = max(2000, min(12000, correction_temp))

    # Tint from G vs R+B average
    gb_ratio = avg_g / ((avg_r + avg_b) / 2)
    correction_tint = int((1.0 - gb_ratio) * 50)
    correction_tint = max(-150, min(150, correction_tint))

    # Nikon WB coefficients
    nikon_r = r_gain
    nikon_b = b_gain

    return {
        'r_gain': r_gain,
        'g_gain': g_gain,
        'b_gain': b_gain,
        'measured_rgb': (int(avg_r), int(avg_g), int(avg_b)),
        'correction_temp': correction_temp,
        'correction_tint': correction_tint,
        'nikon_r_coeff': nikon_r,
        'nikon_b_coeff': nikon_b,
        'rb_ratio': rb_ratio,
    }


def apply_wb_correction(image: np.ndarray, r_gain: float, g_gain: float,
                          b_gain: float) -> np.ndarray:
    """Wendet WB-Korrektur auf ein Bild an."""
    result = image.astype(np.float64)
    result[:, :, 0] *= r_gain
    result[:, :, 1] *= g_gain
    result[:, :, 2] *= b_gain
    return np.clip(result, 0, 255).astype(np.uint8)


def wb_to_xmp_values(wb_result: dict) -> dict:
    """Konvertiert WB-Ergebnis zu Lightroom-Werten."""
    return {
        'Temperature': wb_result['correction_temp'],
        'Tint': wb_result['correction_tint'],
    }


def wb_to_nikon_values(wb_result: dict) -> dict:
    """Konvertiert WB-Ergebnis zu Nikon WB-Koeffizienten."""
    return {
        'r_coeff': wb_result['nikon_r_coeff'],
        'b_coeff': wb_result['nikon_b_coeff'],
    }


# ── Kamera-JPEG Extraktion (#8) ──────────────────────────

def extract_camera_jpeg(raw_path: str) -> Optional[np.ndarray]:
    """
    Extrahiert das eingebettete Kamera-JPEG aus einer RAW-Datei.
    Dieses JPEG zeigt den Look wie die Kamera ihn mit dem
    Picture Control / Film Simulation angewendet hat.
    """
    try:
        import rawpy
        with rawpy.imread(raw_path) as raw:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(thumb.data)).convert('RGB')
                return np.array(img)
    except Exception:
        pass
    return None


def compare_jpeg_vs_raw(raw_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Gibt (camera_jpeg, raw_develop) Tuple zurück.
    camera_jpeg = eingebettetes JPEG mit Kamera-Look
    raw_develop = neutrale RAW-Entwicklung
    """
    camera_jpeg = extract_camera_jpeg(raw_path)
    if camera_jpeg is None:
        return None

    try:
        import rawpy
        with rawpy.imread(raw_path) as raw:
            raw_dev = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=True,
            )
        return (camera_jpeg, raw_dev)
    except Exception:
        return None


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Histogram Matching (#6) - Passt die Farbverteilung des Quellbildes
    an die des Referenzbildes an.

    Args:
        source: Quellbild (wird angepasst)
        reference: Referenzbild (Ziel-Verteilung)

    Returns:
        Angepasstes Bild
    """
    result = np.zeros_like(source)

    for ch in range(3):
        src_vals = source[:, :, ch].ravel()
        ref_vals = reference[:, :, ch].ravel()

        # CDF des Quellbildes
        src_hist, src_bins = np.histogram(src_vals, bins=256, range=(0, 256))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]

        # CDF des Referenzbildes
        ref_hist, ref_bins = np.histogram(ref_vals, bins=256, range=(0, 256))
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # Mapping: Für jeden Quellwert finde den Referenzwert mit ähnlichem CDF
        mapping = np.zeros(256, dtype=np.uint8)
        for src_val in range(256):
            # Finde den Referenzwert mit dem nächsten CDF-Wert
            idx = np.argmin(np.abs(ref_cdf - src_cdf[src_val]))
            mapping[src_val] = idx

        # Anwenden
        result[:, :, ch] = mapping[source[:, :, ch]]

    return result
