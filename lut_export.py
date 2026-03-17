"""
3D LUT Export (.cube Format)

Erzeugt universelle 3D Lookup Tables aus:
- Kanal-Tausch / Mix-Matrizen
- Tonkurven
- Bildstilen (ImageStyle)
- Fuji-Rezepten
- Nikon Picture Controls

.cube-Dateien funktionieren in: DaVinci Resolve, Premiere Pro,
Final Cut Pro, Photoshop, Capture One, OBS, Handy-Apps etc.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


def write_cube_lut(filepath: str, lut_data: np.ndarray, title: str = "DNG Channel Tool LUT",
                   size: int = 0, domain_min: tuple = (0.0, 0.0, 0.0),
                   domain_max: tuple = (1.0, 1.0, 1.0)):
    """
    Schreibt eine 3D LUT im .cube Format.

    Args:
        filepath: Ausgabepfad (.cube)
        lut_data: 3D Array shape (size, size, size, 3) mit RGB-Werten 0.0-1.0
        title: LUT-Titel
        size: LUT-Größe (wird aus lut_data abgeleitet wenn 0)
        domain_min/max: Eingangsbereich
    """
    if size == 0:
        size = lut_data.shape[0]

    with open(filepath, 'w') as f:
        f.write(f"# Created by DNG Channel Tool\n")
        f.write(f"TITLE \"{title}\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write(f"DOMAIN_MIN {domain_min[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f}\n")
        f.write(f"DOMAIN_MAX {domain_max[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f}\n")
        f.write(f"\n")

        # .cube order: R varies fastest, then G, then B
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    rgb = lut_data[r, g, b]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")


def write_1d_cube_lut(filepath: str, lut_r: np.ndarray, lut_g: np.ndarray,
                       lut_b: np.ndarray, title: str = "DNG Channel Tool 1D LUT",
                       size: int = 0):
    """Schreibt eine 1D LUT im .cube Format."""
    if size == 0:
        size = len(lut_r)

    with open(filepath, 'w') as f:
        f.write(f"# Created by DNG Channel Tool\n")
        f.write(f"TITLE \"{title}\"\n")
        f.write(f"LUT_1D_SIZE {size}\n")
        f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n\n")

        for i in range(size):
            f.write(f"{lut_r[i]:.6f} {lut_g[i]:.6f} {lut_b[i]:.6f}\n")


def generate_identity_lut(size: int = 33) -> np.ndarray:
    """Erzeugt eine Identity-LUT (keine Veränderung)."""
    lut = np.zeros((size, size, size, 3), dtype=np.float64)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                lut[r, g, b] = [r / (size - 1), g / (size - 1), b / (size - 1)]
    return lut


def apply_transform_to_lut(lut: np.ndarray, transform_fn: Callable) -> np.ndarray:
    """Wendet eine Transformationsfunktion auf eine LUT an."""
    size = lut.shape[0]
    result = np.zeros_like(lut)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                rgb_in = lut[r, g, b]
                rgb_out = transform_fn(rgb_in)
                result[r, g, b] = np.clip(rgb_out, 0.0, 1.0)
    return result


# ── Konverter ─────────────────────────────────────────────

def mix_matrix_to_lut(mix_matrix: np.ndarray, size: int = 33,
                       title: str = "Channel Mix") -> np.ndarray:
    """Erzeugt 3D LUT aus einer 3x3 Mix-Matrix."""
    lut = generate_identity_lut(size)
    m = mix_matrix.astype(np.float64)

    for r in range(size):
        for g in range(size):
            for b in range(size):
                rgb = lut[r, g, b]
                lut[r, g, b] = np.clip(m @ rgb, 0.0, 1.0)
    return lut


def tone_curve_to_lut(curve_points: List[Tuple[int, int]], size: int = 33,
                       per_channel: bool = False,
                       curve_r: List[Tuple[int, int]] = None,
                       curve_g: List[Tuple[int, int]] = None,
                       curve_b: List[Tuple[int, int]] = None) -> np.ndarray:
    """Erzeugt 3D LUT aus Tonkurven-Punkten."""
    # Interpoliere Kurve zu 256-Einträge LUT
    lum_lut = _interpolate_curve_normalized(curve_points)

    if per_channel and curve_r and curve_g and curve_b:
        r_lut = _interpolate_curve_normalized(curve_r)
        g_lut = _interpolate_curve_normalized(curve_g)
        b_lut = _interpolate_curve_normalized(curve_b)
    else:
        r_lut = g_lut = b_lut = lum_lut

    lut = generate_identity_lut(size)
    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                r_val = lut[ri, gi, bi, 0]
                g_val = lut[ri, gi, bi, 1]
                b_val = lut[ri, gi, bi, 2]

                # Apply curve via interpolation
                lut[ri, gi, bi, 0] = np.interp(r_val, np.linspace(0, 1, 256), r_lut)
                lut[ri, gi, bi, 1] = np.interp(g_val, np.linspace(0, 1, 256), g_lut)
                lut[ri, gi, bi, 2] = np.interp(b_val, np.linspace(0, 1, 256), b_lut)

    return lut


def style_to_lut(style, size: int = 33) -> np.ndarray:
    """Erzeugt 3D LUT aus einem ImageStyle."""
    lut = generate_identity_lut(size)

    # Per-channel curves
    if style.tone_curve_r and style.tone_curve_g and style.tone_curve_b:
        r_lut = _interpolate_curve_normalized(style.tone_curve_r)
        g_lut = _interpolate_curve_normalized(style.tone_curve_g)
        b_lut = _interpolate_curve_normalized(style.tone_curve_b)
    elif style.tone_curve:
        r_lut = g_lut = b_lut = _interpolate_curve_normalized(style.tone_curve)
    else:
        return lut

    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                rv, gv, bv = lut[ri, gi, bi]
                x = np.linspace(0, 1, 256)
                lut[ri, gi, bi, 0] = np.interp(rv, x, r_lut)
                lut[ri, gi, bi, 1] = np.interp(gv, x, g_lut)
                lut[ri, gi, bi, 2] = np.interp(bv, x, b_lut)

    # Monochrome
    if style.is_monochrome:
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = lut[ri, gi, bi]
                    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    lut[ri, gi, bi] = [gray, gray, gray]

    return lut


def fuji_recipe_to_lut(recipe, size: int = 33) -> np.ndarray:
    """Erzeugt 3D LUT aus einem Fuji-Rezept (approximiert)."""
    lut = generate_identity_lut(size)

    # Highlight/Shadow → Tone curve approximation
    h = recipe.highlight  # -2 to +4
    s = recipe.shadow     # -2 to +4

    # Build approximate tone curve from highlight/shadow
    # Highlight affects top half, Shadow affects bottom half
    curve = np.linspace(0, 1, 256)
    # Shadow: positive = lighter shadows, negative = darker
    shadow_boost = s * 0.04
    highlight_boost = h * 0.03
    for i in range(256):
        x = i / 255.0
        if x < 0.5:
            curve[i] = x + shadow_boost * (0.5 - x) * 2
        else:
            curve[i] = x + highlight_boost * (x - 0.5) * 2
    curve = np.clip(curve, 0, 1)

    # Saturation adjustment
    sat_factor = 1.0 + recipe.color * 0.08  # -4..+4 → 0.68..1.32

    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                rv, gv, bv = lut[ri, gi, bi]

                # Apply tone curve
                x = np.linspace(0, 1, 256)
                rv = float(np.interp(rv, x, curve))
                gv = float(np.interp(gv, x, curve))
                bv = float(np.interp(bv, x, curve))

                # Apply saturation
                gray = 0.299 * rv + 0.587 * gv + 0.114 * bv
                rv = gray + (rv - gray) * sat_factor
                gv = gray + (gv - gray) * sat_factor
                bv = gray + (bv - gray) * sat_factor

                lut[ri, gi, bi] = np.clip([rv, gv, bv], 0, 1)

    # Monochrome
    if recipe.is_monochrome:
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = lut[ri, gi, bi]
                    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    lut[ri, gi, bi] = [gray, gray, gray]

    return lut


# ── Kombinations-LUT ─────────────────────────────────────

def combined_lut(mix_matrix: np.ndarray = None,
                 tone_curve: List[Tuple[int, int]] = None,
                 saturation: float = 1.0,
                 monochrome: bool = False,
                 size: int = 33) -> np.ndarray:
    """
    Erzeugt eine kombinierte 3D LUT aus mehreren Transformationen.

    Args:
        mix_matrix: 3x3 Kanal-Mix-Matrix (optional)
        tone_curve: Tonkurven-Punkte (optional)
        saturation: Sättigungsfaktor (1.0 = neutral)
        monochrome: S/W-Konvertierung
        size: LUT-Größe
    """
    lut = generate_identity_lut(size)

    # 1. Channel mix
    if mix_matrix is not None:
        m = mix_matrix.astype(np.float64)
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    lut[ri, gi, bi] = np.clip(m @ lut[ri, gi, bi], 0, 1)

    # 2. Tone curve
    if tone_curve:
        tc = _interpolate_curve_normalized(tone_curve)
        x = np.linspace(0, 1, 256)
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = lut[ri, gi, bi]
                    lut[ri, gi, bi, 0] = np.interp(rgb[0], x, tc)
                    lut[ri, gi, bi, 1] = np.interp(rgb[1], x, tc)
                    lut[ri, gi, bi, 2] = np.interp(rgb[2], x, tc)

    # 3. Saturation
    if saturation != 1.0:
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = lut[ri, gi, bi]
                    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    lut[ri, gi, bi] = np.clip(
                        [gray + (c - gray) * saturation for c in rgb], 0, 1)

    # 4. Monochrome
    if monochrome:
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    rgb = lut[ri, gi, bi]
                    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    lut[ri, gi, bi] = [gray, gray, gray]

    return lut


# ── Helpers ───────────────────────────────────────────────

def _interpolate_curve_normalized(points: List[Tuple[int, int]]) -> np.ndarray:
    """Interpoliert Tonkurven-Punkte (0-255) zu normalisiertem Array (0.0-1.0)."""
    if not points:
        return np.linspace(0, 1, 256)

    pts = sorted(points, key=lambda p: p[0])
    x_pts = np.array([p[0] / 255.0 for p in pts])
    y_pts = np.array([p[1] / 255.0 for p in pts])

    x_out = np.linspace(0, 1, 256)
    y_out = np.interp(x_out, x_pts, y_pts)
    return np.clip(y_out, 0, 1)


def apply_lut_to_image(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Wendet eine 3D LUT auf ein Bild an (für Vorschau)."""
    size = lut.shape[0]
    h, w = image.shape[:2]

    # Normalisiere zu 0-1
    img_f = image.astype(np.float64) / 255.0

    # Trilineare Interpolation
    result = np.zeros_like(img_f)
    scale = size - 1

    for y in range(h):
        for x in range(w):
            r, g, b = img_f[y, x] * scale
            r0, g0, b0 = int(r), int(g), int(b)
            r1 = min(r0 + 1, size - 1)
            g1 = min(g0 + 1, size - 1)
            b1 = min(b0 + 1, size - 1)
            fr, fg, fb = r - r0, g - g0, b - b0

            # Trilinear interpolation
            c000 = lut[r0, g0, b0]
            c100 = lut[r1, g0, b0]
            c010 = lut[r0, g1, b0]
            c110 = lut[r1, g1, b0]
            c001 = lut[r0, g0, b1]
            c101 = lut[r1, g0, b1]
            c011 = lut[r0, g1, b1]
            c111 = lut[r1, g1, b1]

            c00 = c000 * (1 - fr) + c100 * fr
            c01 = c001 * (1 - fr) + c101 * fr
            c10 = c010 * (1 - fr) + c110 * fr
            c11 = c011 * (1 - fr) + c111 * fr

            c0 = c00 * (1 - fg) + c10 * fg
            c1 = c01 * (1 - fg) + c11 * fg

            result[y, x] = c0 * (1 - fb) + c1 * fb

    return np.clip(result * 255, 0, 255).astype(np.uint8)
