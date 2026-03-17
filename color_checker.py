"""
Color Checker Kalibrierung (#5)

Analysiert ein Foto einer Farbkarte (z.B. X-Rite ColorChecker)
und erzeugt ein DCP-Profil oder eine Korrektur-LUT.

Unterstützt:
- X-Rite/Calibrite ColorChecker Classic (24 Felder)
- Manuelle Farbfeld-Auswahl
- Automatische Korrekturmatrix-Berechnung
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# X-Rite ColorChecker Classic 24 Referenzwerte (sRGB, D50)
# Reihenfolge: Zeile 1 (oben, links→rechts), Zeile 2, Zeile 3, Zeile 4
COLORCHECKER_SRGB = [
    # Row 1: Natural colors
    (115, 82, 68),    # Dark Skin
    (194, 150, 130),  # Light Skin
    (98, 122, 157),   # Blue Sky
    (87, 108, 67),    # Foliage
    (133, 128, 177),  # Blue Flower
    (103, 189, 170),  # Bluish Green
    # Row 2: Miscellaneous
    (214, 126, 44),   # Orange
    (80, 91, 166),    # Purplish Blue
    (193, 90, 99),    # Moderate Red
    (94, 60, 108),    # Purple
    (157, 188, 64),   # Yellow Green
    (224, 163, 46),   # Orange Yellow
    # Row 3: Primary/Secondary
    (56, 61, 150),    # Blue
    (70, 148, 73),    # Green
    (175, 54, 60),    # Red
    (231, 199, 31),   # Yellow
    (187, 86, 149),   # Magenta
    (8, 133, 161),    # Cyan
    # Row 4: Grayscale
    (243, 243, 242),  # White
    (200, 200, 200),  # Neutral 8
    (160, 160, 160),  # Neutral 6.5
    (122, 122, 121),  # Neutral 5
    (85, 85, 85),     # Neutral 3.5
    (52, 52, 52),     # Black
]

COLORCHECKER_NAMES = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower", "Bluish Green",
    "Orange", "Purplish Blue", "Moderate Red", "Purple", "Yellow Green", "Orange Yellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan",
    "White", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black",
]


@dataclass
class ColorPatch:
    """Ein gemessenes Farbfeld."""
    name: str
    measured_rgb: Tuple[int, int, int]     # Gemessener Wert aus dem Foto
    reference_rgb: Tuple[int, int, int]    # Referenzwert (soll)


@dataclass
class CalibrationResult:
    """Ergebnis einer Kalibrierung."""
    correction_matrix: np.ndarray    # 3x3 Korrekturmatrix
    patches: List[ColorPatch]
    avg_delta_e: float               # Durchschnittlicher Farbabstand
    max_delta_e: float               # Maximaler Farbabstand
    description: str = ""


def sample_patch_color(image: np.ndarray, center_x: int, center_y: int,
                        patch_size: int = 20) -> Tuple[int, int, int]:
    """
    Misst die durchschnittliche Farbe eines Bereichs im Bild.

    Args:
        image: RGB numpy array
        center_x, center_y: Mittelpunkt des Messbereichs
        patch_size: Größe des Messquadrats (Pixel)
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half)

    patch = image[y1:y2, x1:x2]
    avg = np.mean(patch.reshape(-1, 3), axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def compute_correction_matrix(patches: List[ColorPatch]) -> CalibrationResult:
    """
    Berechnet eine 3x3 Korrekturmatrix aus gemessenen vs. Referenz-Farbfeldern.

    Verwendet Least-Squares-Optimierung:
    reference = M @ measured
    """
    n = len(patches)
    if n < 3:
        raise ValueError("Mindestens 3 Farbfelder benötigt")

    # Build matrices
    measured = np.array([p.measured_rgb for p in patches], dtype=np.float64) / 255.0
    reference = np.array([p.reference_rgb for p in patches], dtype=np.float64) / 255.0

    # Solve: reference = measured @ M^T  →  M^T = (measured^T @ measured)^-1 @ measured^T @ reference
    # Using least squares: M = (reference^T @ measured) @ (measured^T @ measured)^-1
    M, residuals, rank, sv = np.linalg.lstsq(measured, reference, rcond=None)
    correction_matrix = M.T  # 3x3 matrix: corrected = M @ input

    # Delta E berechnen (CIELAB ΔE76)
    corrected = measured @ M

    def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Konvertiert sRGB (0-1) nach CIELAB."""
        # sRGB → Linear RGB
        linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        # Linear RGB → XYZ (D50 Referenz-Weiß)
        m_xyz = np.array([
            [0.4360747, 0.3850649, 0.1430804],
            [0.2225045, 0.7168786, 0.0606169],
            [0.0139322, 0.0971045, 0.7141733],
        ])
        xyz = linear @ m_xyz.T
        # D50 Referenz-Weißpunkt
        xyz_ref = np.array([0.9642, 1.0000, 0.8251])
        xyz_n = xyz / xyz_ref
        # XYZ → Lab
        delta = 6 / 29
        f = np.where(xyz_n > delta ** 3,
                     np.cbrt(xyz_n),
                     xyz_n / (3 * delta ** 2) + 4 / 29)
        L = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b_val = 200 * (f[1] - f[2])
        return np.array([L, a, b_val])

    deltas = []
    for i in range(n):
        lab_ref = _srgb_to_lab(reference[i])
        lab_cor = _srgb_to_lab(np.clip(corrected[i], 0, 1))
        delta = np.sqrt(np.sum((lab_ref - lab_cor) ** 2))
        deltas.append(delta)

    return CalibrationResult(
        correction_matrix=correction_matrix,
        patches=patches,
        avg_delta_e=float(np.mean(deltas)),
        max_delta_e=float(np.max(deltas)),
    )


def auto_detect_colorchecker(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Versucht automatisch die ColorChecker-Felder im Bild zu finden.
    Gibt die Mittelpunkte der 24 Felder zurück.

    Vereinfachter Algorithmus: Sucht nach dem 6x4 Gitter.
    Für Produktionsqualität wäre OpenCV nötig.
    """
    # Vereinfachte Erkennung: Annahme dass ColorChecker das Bild dominiert
    h, w = image.shape[:2]

    # Gitter berechnen (6 Spalten, 4 Zeilen)
    margin_x = w * 0.08
    margin_y = h * 0.08
    usable_w = w - 2 * margin_x
    usable_h = h - 2 * margin_y

    centers = []
    for row in range(4):
        for col in range(6):
            cx = int(margin_x + (col + 0.5) * usable_w / 6)
            cy = int(margin_y + (row + 0.5) * usable_h / 4)
            centers.append((cx, cy))

    return centers


def calibrate_from_colorchecker(image: np.ndarray,
                                  patch_centers: List[Tuple[int, int]] = None,
                                  patch_size: int = 20) -> CalibrationResult:
    """
    Kalibriert anhand eines ColorChecker-Fotos.

    Args:
        image: RGB Foto der Farbkarte
        patch_centers: Mittelpunkte der 24 Felder (oder None für Auto-Erkennung)
        patch_size: Messfeldgröße
    """
    if patch_centers is None:
        patch_centers = auto_detect_colorchecker(image)

    if len(patch_centers) != 24:
        raise ValueError(f"Erwartet 24 Farbfelder, gefunden: {len(patch_centers)}")

    patches = []
    for i, (cx, cy) in enumerate(patch_centers):
        measured = sample_patch_color(image, cx, cy, patch_size)
        reference = COLORCHECKER_SRGB[i]
        patches.append(ColorPatch(
            name=COLORCHECKER_NAMES[i],
            measured_rgb=measured,
            reference_rgb=reference,
        ))

    result = compute_correction_matrix(patches)
    result.description = f"ColorChecker Kalibrierung ({len(patches)} Felder)"
    return result


def calibration_to_dcp(result: CalibrationResult, camera_model: str,
                        profile_name: str = "Kalibriert"):
    """Erzeugt ein DCP-Profil aus dem Kalibrierungsergebnis."""
    from dcp_io import DCPProfile, ILLUMINANT_D65

    profile = DCPProfile(
        camera_model=camera_model,
        profile_name=profile_name,
        color_matrix_1=result.correction_matrix,
        illuminant_1=ILLUMINANT_D65,
        copyright="DNG Channel Tool - Color Checker Calibration",
    )
    return profile


def calibration_to_lut(result: CalibrationResult, size: int = 33) -> np.ndarray:
    """Erzeugt eine 3D LUT aus dem Kalibrierungsergebnis."""
    from lut_export import generate_identity_lut

    lut = generate_identity_lut(size)
    M = result.correction_matrix

    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                rgb = lut[ri, gi, bi]
                lut[ri, gi, bi] = np.clip(M @ rgb, 0, 1)

    return lut
