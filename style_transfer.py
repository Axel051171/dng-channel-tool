"""
Bildstil-Analyse & Übertragung

Extrahiert Farbcharakteristiken aus Bildern und erzeugt daraus
Nikon Picture Controls, Adobe XMP-Presets und Tonkurven.

Methoden:
1. Einzelbild-Analyse: Tonkurve + Farbstatistiken aus einem Bild
2. Paar-Vergleich: Original + Referenz vergleichen → Transformation berechnen
3. Stil-Transfer: Farbverteilung eines Bildes auf ein anderes übertragen
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from PIL import Image


@dataclass
class ImageStyle:
    """Extrahierter Bildstil mit allen relevanten Parametern."""
    name: str = "Extrahierter Stil"

    # Tone curve (input, output) pairs 0-255
    tone_curve: List[Tuple[int, int]] = field(default_factory=list)

    # Per-channel curves
    tone_curve_r: List[Tuple[int, int]] = field(default_factory=list)
    tone_curve_g: List[Tuple[int, int]] = field(default_factory=list)
    tone_curve_b: List[Tuple[int, int]] = field(default_factory=list)

    # Statistics (Lightroom-compatible range)
    contrast: int = 0          # -100 to +100
    brightness: int = 0        # -100 to +100
    saturation: int = 0        # -100 to +100
    hue_shift: int = 0         # -180 to +180
    clarity: int = 0           # -100 to +100

    # Color temperature hint
    temperature: int = 0       # Kelvin (0 = neutral)
    tint: int = 0              # Green-Magenta

    # Detected characteristics
    is_monochrome: bool = False
    is_high_contrast: bool = False
    is_faded: bool = False     # Lifted blacks
    black_point: int = 0       # 0-255
    white_point: int = 255     # 0-255

    # Source info
    source_file: str = ""


def analyze_image(image: np.ndarray, name: str = "Stil") -> ImageStyle:
    """
    Analysiert ein Bild und extrahiert seinen Farbstil.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        name: Name für den Stil

    Returns:
        ImageStyle mit extrahierten Parametern
    """
    style = ImageStyle(name=name)
    h, w = image.shape[:2]

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # ── Luminanz berechnen ──
    lum = (0.299 * r.astype(float) + 0.587 * g.astype(float) +
           0.114 * b.astype(float))

    # ── Monochrom-Erkennung ──
    color_std = np.std(r.astype(float) - g.astype(float)) + \
                np.std(g.astype(float) - b.astype(float))
    style.is_monochrome = color_std < 5.0

    # ── Tonkurve aus Luminanz-Histogramm ableiten ──
    style.tone_curve = _extract_tone_curve(lum)
    style.tone_curve_r = _extract_tone_curve(r.astype(float))
    style.tone_curve_g = _extract_tone_curve(g.astype(float))
    style.tone_curve_b = _extract_tone_curve(b.astype(float))

    # ── Schwarz- und Weißpunkt ──
    lum_flat = lum.ravel()
    style.black_point = int(np.percentile(lum_flat, 0.5))
    style.white_point = int(np.percentile(lum_flat, 99.5))
    style.is_faded = style.black_point > 15

    # ── Kontrast ──
    std_lum = np.std(lum_flat)
    # Normaler Kontrast liegt bei ca. std=50-60
    style.contrast = int(np.clip((std_lum - 55) * 2, -100, 100))
    style.is_high_contrast = style.contrast > 30

    # ── Helligkeit ──
    mean_lum = np.mean(lum_flat)
    style.brightness = int(np.clip((mean_lum - 128) * 100 / 128, -100, 100))

    # ── Sättigung ──
    if not style.is_monochrome:
        # HSV-basierte Sättigungsanalyse
        max_rgb = np.maximum(np.maximum(r, g), b).astype(float)
        min_rgb = np.minimum(np.minimum(r, g), b).astype(float)
        delta = max_rgb - min_rgb
        sat_map = np.where(max_rgb > 0, delta / max_rgb, 0)
        mean_sat = np.mean(sat_map)
        # Normale Sättigung ~0.3-0.4
        style.saturation = int(np.clip((mean_sat - 0.35) * 300, -100, 100))
    else:
        style.saturation = -100

    # ── Farbtemperatur-Schätzung ──
    if not style.is_monochrome:
        mean_r = np.mean(r.astype(float))
        mean_b = np.mean(b.astype(float))
        # Warme Bilder: R > B, Kalte: B > R
        rb_ratio = mean_r / max(mean_b, 1)
        if rb_ratio > 1.1:
            style.temperature = int((rb_ratio - 1.0) * 3000)  # Warm
        elif rb_ratio < 0.9:
            style.temperature = int((rb_ratio - 1.0) * 3000)  # Cold

    # ── Klarheit (lokaler Kontrast) ──
    # Vereinfachte Schätzung über Hochpass-Energie
    from PIL import ImageFilter
    pil_img = Image.fromarray(image).convert('L')
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=10))
    detail = np.array(pil_img).astype(float) - np.array(blurred).astype(float)
    detail_energy = np.std(detail)
    style.clarity = int(np.clip((detail_energy - 20) * 3, -100, 100))

    return style


def compare_images(original: np.ndarray, styled: np.ndarray,
                   name: str = "Übertragener Stil") -> ImageStyle:
    """
    Vergleicht Original und gestyltes Bild, berechnet die Transformation.

    Args:
        original: Originalbild (RGB uint8)
        styled: Gestyltes Bild (RGB uint8)
        name: Name für den Stil

    Returns:
        ImageStyle der die Transformation beschreibt
    """
    style = ImageStyle(name=name)

    # Bilder auf gleiche Größe bringen
    h = min(original.shape[0], styled.shape[0])
    w = min(original.shape[1], styled.shape[1])
    orig = original[:h, :w].astype(float)
    styl = styled[:h, :w].astype(float)

    # ── Tonkurve durch Histogramm-Mapping ──
    # Für jeden Eingangswert: Was ist der durchschnittliche Ausgangswert?
    orig_lum = (0.299 * orig[:, :, 0] + 0.587 * orig[:, :, 1] +
                0.114 * orig[:, :, 2])
    styl_lum = (0.299 * styl[:, :, 0] + 0.587 * styl[:, :, 1] +
                0.114 * styl[:, :, 2])

    style.tone_curve = _compute_transfer_curve(orig_lum, styl_lum)

    # Per-channel curves
    style.tone_curve_r = _compute_transfer_curve(orig[:, :, 0], styl[:, :, 0])
    style.tone_curve_g = _compute_transfer_curve(orig[:, :, 1], styl[:, :, 1])
    style.tone_curve_b = _compute_transfer_curve(orig[:, :, 2], styl[:, :, 2])

    # ── Sättigungsänderung ──
    orig_style = analyze_image(original[:h, :w], "orig")
    styl_style = analyze_image(styled[:h, :w], "styl")

    style.contrast = styl_style.contrast - orig_style.contrast
    style.brightness = styl_style.brightness - orig_style.brightness
    style.saturation = styl_style.saturation - orig_style.saturation
    style.clarity = styl_style.clarity - orig_style.clarity
    style.is_monochrome = styl_style.is_monochrome
    style.black_point = styl_style.black_point
    style.white_point = styl_style.white_point
    style.is_faded = styl_style.is_faded
    style.is_high_contrast = styl_style.is_high_contrast
    style.temperature = styl_style.temperature

    return style


def apply_style(image: np.ndarray, style: ImageStyle) -> np.ndarray:
    """Wendet einen extrahierten Stil auf ein Bild an."""
    result = image.copy().astype(float)

    # Tonkurve anwenden (per Kanal wenn verfügbar)
    if style.tone_curve_r and style.tone_curve_g and style.tone_curve_b:
        lut_r = _curve_to_lut(style.tone_curve_r)
        lut_g = _curve_to_lut(style.tone_curve_g)
        lut_b = _curve_to_lut(style.tone_curve_b)
        result[:, :, 0] = lut_r[image[:, :, 0]]
        result[:, :, 1] = lut_g[image[:, :, 1]]
        result[:, :, 2] = lut_b[image[:, :, 2]]
    elif style.tone_curve:
        lut = _curve_to_lut(style.tone_curve)
        for ch in range(3):
            result[:, :, ch] = lut[image[:, :, ch]]

    # Monochrom
    if style.is_monochrome:
        lum = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        result[:, :, 0] = lum
        result[:, :, 1] = lum
        result[:, :, 2] = lum

    return np.clip(result, 0, 255).astype(np.uint8)


def style_to_xmp(style: ImageStyle, filepath: str, camera_model: str = ""):
    """Exportiert einen ImageStyle als Lightroom XMP-Preset."""
    import uuid

    curve_items = []
    for x, y in style.tone_curve:
        curve_items.append(f'    <rdf:li>{x}, {y}</rdf:li>')
    curve_xml = "\n".join(curve_items) if curve_items else '    <rdf:li>0, 0</rdf:li>\n    <rdf:li>255, 255</rdf:li>'

    # Per-channel curves
    channel_curves = ""
    for ch_name, ch_curve in [("Red", style.tone_curve_r),
                               ("Green", style.tone_curve_g),
                               ("Blue", style.tone_curve_b)]:
        if ch_curve:
            items = "\n".join(f'    <rdf:li>{x}, {y}</rdf:li>' for x, y in ch_curve)
            channel_curves += f"""
   <crs:ToneCurvePV2012{ch_name}>
    <rdf:Seq>
{items}
    </rdf:Seq>
   </crs:ToneCurvePV2012{ch_name}>"""

    grayscale = "True" if style.is_monochrome else "False"
    cam_restrict = f'crs:CameraModelRestriction="{camera_model}"' if camera_model else ""

    xmp = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - Style Transfer">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{str(uuid.uuid4()).upper()}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    {cam_restrict}
    crs:Copyright="DNG Channel Tool - Style Transfer"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:Contrast="{style.contrast}"
    crs:Saturation="{style.saturation}"
    crs:Clarity="{style.clarity}"
    crs:ConvertToGrayscale="{grayscale}"
    crs:ToneCurveName2012="Custom"
    crs:Group{{Name}}="Extrahierte Stile"
    >
   <crs:ToneCurvePV2012>
    <rdf:Seq>
{curve_xml}
    </rdf:Seq>
   </crs:ToneCurvePV2012>{channel_curves}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp)


def style_to_nikon_pc(style: ImageStyle):
    """Konvertiert einen ImageStyle zu NikonPictureControlFile-Parametern."""
    from npc_io import NikonPictureControlFile

    pc = NikonPictureControlFile()
    pc.name = style.name[:19]
    pc.base = "MONOCHROME" if style.is_monochrome else "STANDARD"

    # LR range (-100..+100) → Nikon range (0..255, center=128)
    def lr_to_nikon(val, scale=0.5):
        return max(0, min(255, int(128 + val * scale)))

    pc.contrast = lr_to_nikon(style.contrast)
    pc.brightness = lr_to_nikon(style.brightness)
    pc.saturation = lr_to_nikon(style.saturation)
    pc.clarity = lr_to_nikon(style.clarity)
    pc.tone_curve = style.tone_curve if style.tone_curve else [(0, 0), (255, 255)]

    return pc


# ── Internal Helpers ──────────────────────────────────────

def _extract_tone_curve(channel: np.ndarray, num_points: int = 9) -> list:
    """
    Extrahiert eine Tonkurve aus einem einzelnen Kanal.
    Nutzt kumulative Histogramm-Analyse.
    """
    flat = channel.ravel()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    cdf = np.cumsum(hist).astype(float)
    cdf_norm = cdf / cdf[-1] * 255

    # Sample an gleichmäßig verteilten Eingangswerten
    points = []
    step = 255 / (num_points - 1)
    for i in range(num_points):
        x = int(i * step)
        y = int(np.clip(cdf_norm[min(x, 255)], 0, 255))
        points.append((x, y))

    # Vereinfache: entferne Punkte die nahe an der Diagonale liegen
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        x, y = points[i]
        # Abweichung von der Diagonale
        if abs(y - x) > 3:
            simplified.append((x, y))
    simplified.append(points[-1])

    return simplified


def _compute_transfer_curve(orig_ch: np.ndarray, styled_ch: np.ndarray,
                             num_points: int = 12) -> list:
    """
    Berechnet die Transfer-Kurve zwischen Original und gestyltem Kanal.
    Für jeden Eingangswert: durchschnittlicher Ausgangswert.
    """
    orig_flat = np.clip(orig_ch.ravel(), 0, 255).astype(int)
    styl_flat = np.clip(styled_ch.ravel(), 0, 255).astype(int)

    # Für jeden Eingangswert den durchschnittlichen Ausgangswert berechnen
    sums = np.zeros(256, dtype=float)
    counts = np.zeros(256, dtype=float)

    for o, s in zip(orig_flat, styl_flat):
        sums[o] += s
        counts[o] += 1

    # Transfer function
    transfer = np.zeros(256)
    for i in range(256):
        if counts[i] > 0:
            transfer[i] = sums[i] / counts[i]
        else:
            transfer[i] = i  # Identity fallback

    # Glätten
    kernel = np.ones(5) / 5
    transfer = np.convolve(transfer, kernel, mode='same')

    # Sample Kontrollpunkte
    points = []
    step = 255 / (num_points - 1)
    for i in range(num_points):
        x = int(i * step)
        y = int(np.clip(transfer[x], 0, 255))
        points.append((x, y))

    return points


def _curve_to_lut(points: list) -> np.ndarray:
    """Konvertiert Tonkurven-Punkte zu einer 256-Einträge LUT."""
    lut = np.zeros(256, dtype=float)
    pts = sorted(points, key=lambda p: p[0])

    for i in range(256):
        if i <= pts[0][0]:
            lut[i] = pts[0][1]
        elif i >= pts[-1][0]:
            lut[i] = pts[-1][1]
        else:
            for j in range(len(pts) - 1):
                if pts[j][0] <= i <= pts[j + 1][0]:
                    t = (i - pts[j][0]) / max(1, pts[j + 1][0] - pts[j][0])
                    lut[i] = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
                    break

    return np.clip(lut, 0, 255).astype(np.uint8)
