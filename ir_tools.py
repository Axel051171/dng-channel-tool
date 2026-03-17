"""
Infrarot-Fotografie Werkzeuge

Spezialisierte Tools für IR-konvertierte Kameras:
1. IR-Weißabgleich-Assistent (Vegetation-basiert)
2. False-Color Preset-Sammlung (Aerochrome, Blue Sky, Goldie, etc.)
3. IR-Filter-Simulation (590nm - 850nm)
4. Hotspot-Erkennung und Korrektur
5. Custom DCP-Profile pro Filter-Typ
6. NDVI-Berechnung (Vegetationsindex)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ── IR Filter Wellenlängen ────────────────────────────────

@dataclass
class IRFilter:
    """Definition eines IR-Cutoff-Filters."""
    name: str
    cutoff_nm: int        # Cutoff-Wellenlänge in nm
    description: str
    # Kanalgewichte für Simulation [R, G, B]
    # IR-Filter lassen hauptsächlich Rot durch, je nach Cutoff
    channel_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    # Typische WB-Korrektur (Temp, Tint)
    typical_wb_temp: int = 2200
    typical_wb_tint: int = -50
    # Relative Empfindlichkeit pro Kanal bei diesem Filter
    sensitivity: Tuple[float, float, float] = (1.0, 0.5, 0.1)


IR_FILTERS = {
    "590nm": IRFilter(
        name="590nm (Deep Yellow/Orange)",
        cutoff_nm=590,
        description="Lässt sichtbares Rot + IR durch. Maximale Farbvielfalt.",
        channel_weights=(1.0, 0.6, 0.15),
        sensitivity=(1.0, 0.7, 0.2),
        typical_wb_temp=2500, typical_wb_tint=-30,
    ),
    "630nm": IRFilter(
        name="630nm (Deep Red)",
        cutoff_nm=630,
        description="Weniger sichtbares Licht, gute Balance Farbe/IR-Effekt.",
        channel_weights=(1.0, 0.4, 0.08),
        sensitivity=(1.0, 0.5, 0.12),
        typical_wb_temp=2300, typical_wb_tint=-40,
    ),
    "680nm": IRFilter(
        name="680nm (Standard IR)",
        cutoff_nm=680,
        description="Klassischer IR-Filter. Guter Wood-Effekt.",
        channel_weights=(1.0, 0.25, 0.03),
        sensitivity=(1.0, 0.35, 0.06),
        typical_wb_temp=2200, typical_wb_tint=-50,
    ),
    "720nm": IRFilter(
        name="720nm (Standard IR)",
        cutoff_nm=720,
        description="Der populärste IR-Filter. Starker Wood-Effekt, wenig Farbe.",
        channel_weights=(1.0, 0.15, 0.01),
        sensitivity=(1.0, 0.2, 0.03),
        typical_wb_temp=2100, typical_wb_tint=-60,
    ),
    "850nm": IRFilter(
        name="850nm (Deep IR)",
        cutoff_nm=850,
        description="Fast reines IR. Nur S/W möglich, sehr starker Kontrast.",
        channel_weights=(1.0, 0.02, 0.0),
        sensitivity=(1.0, 0.05, 0.01),
        typical_wb_temp=2000, typical_wb_tint=-80,
    ),
    "Full Spectrum": IRFilter(
        name="Full Spectrum (Kein Filter)",
        cutoff_nm=0,
        description="Kamera ohne internen IR-Sperrfilter, externes Filter nötig.",
        channel_weights=(1.0, 1.0, 1.0),
        sensitivity=(1.0, 1.0, 1.0),
        typical_wb_temp=5500, typical_wb_tint=0,
    ),
}


# ── IR False-Color Presets ────────────────────────────────

@dataclass
class IRFalseColorPreset:
    """Ein IR False-Color Preset mit komplettem Pipeline."""
    name: str
    description: str
    # Channel mix matrix (3x3)
    mix_matrix: np.ndarray
    # WB correction (temp, tint relative)
    wb_temp_shift: int = 0
    wb_tint_shift: int = 0
    # Tone adjustments
    contrast: int = 0         # -100 to +100
    saturation: float = 1.0   # Multiplikator
    brightness: int = 0
    # Tone curve points (optional)
    tone_curve: List[Tuple[int, int]] = field(default_factory=list)
    # Is monochrome output
    monochrome: bool = False
    # Recommended filter
    recommended_filter: str = "720nm"


IR_FALSE_COLOR_PRESETS: Dict[str, IRFalseColorPreset] = {
    "Classic Blue Sky": IRFalseColorPreset(
        name="Classic Blue Sky",
        description="Der Klassiker: R↔B Tausch. Himmel wird blau, Vegetation weiß/gelb.",
        mix_matrix=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        wb_temp_shift=-500,
        saturation=1.3,
        recommended_filter="720nm",
    ),

    "Goldie": IRFalseColorPreset(
        name="Goldie",
        description="Warme Goldtöne. Vegetation in Gold/Gelb, Himmel dunkelblau.",
        mix_matrix=np.array([
            [0.8, 0.2, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]),
        wb_temp_shift=500,
        saturation=1.4,
        contrast=10,
        recommended_filter="590nm",
    ),

    "Super Color IR": IRFalseColorPreset(
        name="Super Color IR",
        description="Maximale Farbsättigung. Surreale, psychedelische Farben.",
        mix_matrix=np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        saturation=1.8,
        contrast=15,
        recommended_filter="590nm",
    ),

    "Kodak Aerochrome": IRFalseColorPreset(
        name="Kodak Aerochrome",
        description="Simulation des legendären Kodak IR-Farbfilms. "
                    "Vegetation wird magenta/pink, Himmel cyan/blau.",
        mix_matrix=np.array([
            [0.0, 1.0, 0.0],   # R_out = G_in (IR → rot/magenta)
            [0.0, 0.0, 1.0],   # G_out = B_in
            [1.0, 0.0, 0.0],   # B_out = R_in
        ]),
        saturation=1.5,
        contrast=10,
        wb_temp_shift=-300,
        recommended_filter="590nm",
    ),

    "Chocolate IR": IRFalseColorPreset(
        name="Chocolate IR",
        description="Warme Schokoladentöne. Nostalgischer Look.",
        mix_matrix=np.array([
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
        ]),
        saturation=0.6,
        contrast=20,
        wb_temp_shift=800,
        tone_curve=[(0, 10), (64, 50), (128, 130), (192, 210), (255, 245)],
        recommended_filter="720nm",
    ),

    "S/W IR Kontrast": IRFalseColorPreset(
        name="S/W IR Kontrast",
        description="Klassisches S/W-Infrarot mit starkem Kontrast. "
                    "Dunkler Himmel, helle Vegetation.",
        mix_matrix=np.eye(3),
        monochrome=True,
        contrast=40,
        tone_curve=[(0, 0), (40, 5), (80, 40), (160, 200), (220, 245), (255, 255)],
        recommended_filter="720nm",
    ),

    "Dream IR": IRFalseColorPreset(
        name="Dream IR",
        description="Verträumter, weicher IR-Look mit angehobenen Schwarztönen.",
        mix_matrix=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        saturation=0.8,
        contrast=-15,
        brightness=10,
        tone_curve=[(0, 25), (64, 75), (128, 140), (192, 210), (255, 240)],
        recommended_filter="720nm",
    ),

    "Candy IR": IRFalseColorPreset(
        name="Candy IR",
        description="Bonbon-Farben. Rosa Vegetation, türkiser Himmel.",
        mix_matrix=np.array([
            [0.3, 0.0, 0.7],
            [0.0, 0.8, 0.2],
            [0.7, 0.3, 0.0],
        ]),
        saturation=1.6,
        contrast=5,
        recommended_filter="590nm",
    ),
}


# ── 1. IR-Weißabgleich-Assistent ─────────────────────────

def calculate_ir_wb(image: np.ndarray, x: int, y: int,
                     filter_type: str = "720nm",
                     sample_size: int = 30) -> dict:
    """
    Berechnet den optimalen Weißabgleich für IR-Fotos.

    Klicke auf VEGETATION (die in IR weiß/hell erscheinen soll).
    Der Algorithmus korrigiert so, dass diese Stelle neutral-weiß wird.

    Args:
        image: IR-Foto (RGB, uint8)
        x, y: Klick auf Vegetation
        filter_type: Verwendeter IR-Filter
        sample_size: Messfeldgröße

    Returns:
        dict mit WB-Korrekturdaten
    """
    h, w = image.shape[:2]
    half = sample_size // 2
    patch = image[max(0,y-half):min(h,y+half), max(0,x-half):min(w,x+half)]

    avg_r = float(np.mean(patch[:, :, 0]))
    avg_g = float(np.mean(patch[:, :, 1]))
    avg_b = float(np.mean(patch[:, :, 2]))

    # Ziel: Vegetation soll neutral-weiß werden
    target = max(avg_r, avg_g, avg_b)
    r_gain = target / max(avg_r, 1)
    g_gain = target / max(avg_g, 1)
    b_gain = target / max(avg_b, 1)

    ir_filter = IR_FILTERS.get(filter_type, IR_FILTERS["720nm"])

    # Lightroom-Werte schätzen
    # Bei IR ist die Farbtemperatur extrem niedrig (2000-2500K)
    lr_temp = ir_filter.typical_wb_temp
    lr_tint = ir_filter.typical_wb_tint

    # Feinkorrektur basierend auf der Messung
    rb_ratio = avg_r / max(avg_b, 1)
    if rb_ratio > 2.0:
        lr_temp -= 200
    elif rb_ratio < 1.5:
        lr_temp += 200

    return {
        'r_gain': r_gain,
        'g_gain': g_gain,
        'b_gain': b_gain,
        'measured_rgb': (int(avg_r), int(avg_g), int(avg_b)),
        'lr_temperature': lr_temp,
        'lr_tint': lr_tint,
        'nikon_r_coeff': r_gain / g_gain,
        'nikon_b_coeff': b_gain / g_gain,
        'filter': filter_type,
        'filter_info': ir_filter.description,
    }


def apply_ir_wb(image: np.ndarray, wb: dict) -> np.ndarray:
    """Wendet IR-Weißabgleich auf ein Bild an."""
    result = image.astype(np.float64)
    result[:, :, 0] *= wb['r_gain']
    result[:, :, 1] *= wb['g_gain']
    result[:, :, 2] *= wb['b_gain']
    return np.clip(result, 0, 255).astype(np.uint8)


# ── 2. False-Color Anwendung ─────────────────────────────

def apply_ir_preset(image: np.ndarray, preset: IRFalseColorPreset) -> np.ndarray:
    """Wendet ein IR False-Color Preset auf ein Bild an."""
    img = image.astype(np.float64) / 255.0

    # 1. Channel mix
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    M = preset.mix_matrix
    for ch in range(3):
        result[:, :, ch] = (M[ch, 0] * img[:, :, 0] +
                             M[ch, 1] * img[:, :, 1] +
                             M[ch, 2] * img[:, :, 2])

    # 2. Saturation
    if preset.saturation != 1.0:
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        for ch in range(3):
            result[:, :, ch] = gray + (result[:, :, ch] - gray) * preset.saturation

    # 3. Contrast
    if preset.contrast != 0:
        factor = (100 + preset.contrast) / 100.0
        result = 0.5 + (result - 0.5) * factor

    # 4. Brightness
    if preset.brightness != 0:
        result += preset.brightness / 255.0

    # 5. Tone curve
    if preset.tone_curve:
        lut = _curve_to_lut(preset.tone_curve)
        result_u8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        for ch in range(3):
            result_u8[:, :, ch] = lut[result_u8[:, :, ch]]
        result = result_u8.astype(np.float64) / 255.0

    # 6. Monochrome
    if preset.monochrome:
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        result[:, :, 0] = gray
        result[:, :, 1] = gray
        result[:, :, 2] = gray

    return np.clip(result * 255, 0, 255).astype(np.uint8)


# ── 3. IR-Filter-Simulation ──────────────────────────────

def simulate_ir_filter(image: np.ndarray, filter_type: str = "720nm") -> np.ndarray:
    """
    Simuliert wie ein normales Foto mit einem IR-Filter aussehen würde.

    Nutzt den Rot-Kanal (enthält am meisten IR-Information im sichtbaren Spektrum)
    und gewichtet die Kanäle entsprechend dem Filter-Cutoff.
    """
    ir_filter = IR_FILTERS.get(filter_type, IR_FILTERS["720nm"])
    w = ir_filter.channel_weights

    img = image.astype(np.float64)

    # Gewichtete Summe der Kanäle
    ir_channel = (w[0] * img[:, :, 0] + w[1] * img[:, :, 1] + w[2] * img[:, :, 2])
    total_weight = sum(w)
    if total_weight > 0:
        ir_channel /= total_weight

    # Simulation: R-Kanal dominant (IR), G und B gedämpft
    result = np.zeros_like(img)
    s = ir_filter.sensitivity
    result[:, :, 0] = ir_channel * s[0]
    result[:, :, 1] = ir_channel * s[1]
    result[:, :, 2] = ir_channel * s[2]

    # Normalisieren für sichtbaren Kontrast
    result = result / max(result.max(), 1) * 255

    return np.clip(result, 0, 255).astype(np.uint8)


# ── 4. Hotspot-Erkennung ─────────────────────────────────

@dataclass
class HotspotResult:
    """Ergebnis der Hotspot-Analyse."""
    has_hotspot: bool
    severity: float          # 0.0 (kein) bis 1.0 (stark)
    center_x: int
    center_y: int
    radius: int              # Geschätzter Radius in Pixeln
    brightness_center: float # Durchschnittliche Helligkeit im Zentrum
    brightness_edge: float   # Durchschnittliche Helligkeit am Rand
    description: str


def detect_hotspot(image: np.ndarray, threshold: float = 0.15) -> HotspotResult:
    """
    Erkennt IR-Hotspots (helle Flecken in der Bildmitte).

    Ein Hotspot entsteht wenn das Objektiv IR-Licht intern reflektiert.
    Typisch: Heller kreisförmiger Bereich in der Bildmitte.

    Args:
        image: IR-Foto (RGB uint8)
        threshold: Schwellwert für Hotspot-Erkennung (0.0-1.0)
    """
    h, w = image.shape[:2]
    gray = np.mean(image.astype(np.float64), axis=2)

    # Zentrum vs. Rand vergleichen
    cy, cx = h // 2, w // 2

    # Kreisförmige Messbereiche
    radii = [min(h, w) // 8, min(h, w) // 4, min(h, w) // 2.5]

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    # Innerer Bereich (Zentrum)
    inner_mask = dist < radii[0]
    inner_brightness = np.mean(gray[inner_mask])

    # Mittlerer Ring
    mid_mask = (dist > radii[0]) & (dist < radii[1])
    mid_brightness = np.mean(gray[mid_mask]) if np.any(mid_mask) else inner_brightness

    # Äußerer Ring (Rand)
    outer_mask = dist > radii[2]
    outer_brightness = np.mean(gray[outer_mask]) if np.any(outer_mask) else mid_brightness

    # Hotspot-Stärke: Wie viel heller ist das Zentrum vs. der Rand?
    if outer_brightness > 0:
        brightness_ratio = inner_brightness / outer_brightness - 1.0
    else:
        brightness_ratio = 0

    has_hotspot = brightness_ratio > threshold
    severity = min(1.0, max(0.0, brightness_ratio / 0.5))

    # Hotspot-Radius schätzen
    hotspot_radius = 0
    if has_hotspot:
        for r in range(10, min(h, w) // 2, 5):
            ring = (dist > r - 5) & (dist < r + 5)
            ring_bright = np.mean(gray[ring]) if np.any(ring) else 0
            if ring_bright < inner_brightness * 0.9:
                hotspot_radius = r
                break

    if has_hotspot:
        desc = (f"Hotspot erkannt! Stärke: {severity:.0%}\n"
                f"Zentrum ist {brightness_ratio:.0%} heller als der Rand.\n"
                f"Geschätzter Radius: ~{hotspot_radius}px\n\n"
                f"Empfehlung: Andere Blende oder anderes Objektiv verwenden.")
    else:
        desc = "Kein Hotspot erkannt. Das Objektiv scheint IR-kompatibel zu sein."

    return HotspotResult(
        has_hotspot=has_hotspot,
        severity=severity,
        center_x=cx, center_y=cy,
        radius=hotspot_radius,
        brightness_center=inner_brightness,
        brightness_edge=outer_brightness,
        description=desc,
    )


def correct_hotspot(image: np.ndarray, hotspot: HotspotResult,
                      strength: float = 1.0) -> np.ndarray:
    """
    Korrigiert einen Hotspot durch radiale Abdunkelung.
    """
    if not hotspot.has_hotspot or hotspot.radius < 10:
        return image.copy()

    h, w = image.shape[:2]
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - hotspot.center_x) ** 2 +
                    (y_grid - hotspot.center_y) ** 2)

    # Korrekturmaske: Im Zentrum abdunkeln
    radius = hotspot.radius * 1.5
    correction = np.ones((h, w), dtype=np.float64)

    mask = dist < radius
    # Gauss-förmige Abdunkelung
    factor = hotspot.severity * strength
    correction[mask] = 1.0 - factor * np.exp(-0.5 * (dist[mask] / (radius * 0.5)) ** 2)
    # Innen stärker korrigieren
    inner = dist < hotspot.radius * 0.5
    correction[inner] = 1.0 - factor * 0.8

    result = image.astype(np.float64)
    for ch in range(3):
        result[:, :, ch] *= correction

    return np.clip(result, 0, 255).astype(np.uint8)


# ── 5. Custom DCP für IR-Filter ──────────────────────────

def generate_ir_dcp(camera_model: str, filter_type: str = "720nm",
                      channel_swap: str = "RGB"):
    """
    Generiert ein kameraspecifisches DCP-Profil für IR-Fotografie.

    Args:
        camera_model: EXIF-Kameraname
        filter_type: IR-Filter Typ
        channel_swap: "RGB" (kein Swap), "BGR" (R↔B), etc.
    """
    from dcp_io import DCPProfile, ILLUMINANT_D65
    from channel_swap import ChannelMapping, swap_color_matrix

    ir_filter = IR_FILTERS.get(filter_type, IR_FILTERS["720nm"])

    # Basis-Farbmatrix für IR (angepasst an IR-Sensitivität)
    s = ir_filter.sensitivity
    # IR-angepasste Farbmatrix: Verstärkt den dominanten Kanal
    ir_color_matrix = np.array([
        [1.0 / s[0],  0.0,          0.0],
        [0.0,         1.0 / s[1],   0.0],
        [0.0,         0.0,          1.0 / s[2]],
    ])

    # Normalisieren
    for i in range(3):
        row_sum = np.abs(ir_color_matrix[i]).sum()
        if row_sum > 0:
            ir_color_matrix[i] /= row_sum

    # Channel swap anwenden
    if channel_swap != "RGB" and len(channel_swap) == 3:
        perm = tuple("RGB".index(c) for c in channel_swap)
        mapping = ChannelMapping.from_permutation(perm)
        ir_color_matrix = swap_color_matrix(ir_color_matrix, mapping)

    profile_name = f"IR {filter_type} {channel_swap}"

    profile = DCPProfile(
        camera_model=camera_model,
        profile_name=profile_name,
        color_matrix_1=ir_color_matrix,
        illuminant_1=ILLUMINANT_D65,
        copyright="DNG Channel Tool - IR Profile",
    )
    return profile


# ── 6. NDVI-Berechnung ───────────────────────────────────

def calculate_ndvi(image: np.ndarray, ir_channel: int = 0,
                     vis_channel: int = 2) -> np.ndarray:
    """
    Berechnet den Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - VIS) / (NIR + VIS)

    Bei IR-konvertierten Kameras:
    - NIR ist typischerweise im Rot-Kanal (nach Filter)
    - VIS ist im Blau-Kanal (am wenigsten IR-kontaminiert)

    Args:
        image: IR-Foto (RGB uint8)
        ir_channel: Kanal mit IR-Information (0=R, 1=G, 2=B)
        vis_channel: Kanal mit sichtbarem Licht (0=R, 1=G, 2=B)

    Returns:
        NDVI-Bild als RGB (uint8), farbkodiert:
        - Rot: Kein Pflanzenwuchs (NDVI < 0)
        - Gelb: Wenig Vegetation
        - Grün: Gesunde Vegetation (NDVI > 0.3)
    """
    nir = image[:, :, ir_channel].astype(np.float64)
    vis = image[:, :, vis_channel].astype(np.float64)

    # NDVI berechnen (-1 bis +1)
    denominator = nir + vis
    ndvi = np.where(denominator > 0, (nir - vis) / denominator, 0)

    # Farbkodierung
    result = np.zeros((*ndvi.shape, 3), dtype=np.uint8)

    # Rot: NDVI < 0 (kein Pflanzenwuchs, Wasser, Gebäude)
    neg_mask = ndvi < 0
    result[neg_mask, 0] = np.clip(-ndvi[neg_mask] * 255, 0, 255).astype(np.uint8)

    # Gelb bis Grün: NDVI 0 bis 1
    pos_mask = ndvi >= 0
    # Rot-Anteil nimmt ab
    result[pos_mask, 0] = np.clip((1 - ndvi[pos_mask] * 2) * 200, 0, 200).astype(np.uint8)
    # Grün-Anteil nimmt zu
    result[pos_mask, 1] = np.clip(ndvi[pos_mask] * 255, 0, 255).astype(np.uint8)
    # Blau für sehr gesunde Vegetation
    very_green = ndvi > 0.5
    result[very_green, 2] = np.clip((ndvi[very_green] - 0.5) * 200, 0, 100).astype(np.uint8)

    return result


def ndvi_statistics(image: np.ndarray, ir_channel: int = 0,
                      vis_channel: int = 2) -> dict:
    """Berechnet NDVI-Statistiken für ein Bild."""
    nir = image[:, :, ir_channel].astype(np.float64)
    vis = image[:, :, vis_channel].astype(np.float64)

    denom = nir + vis
    ndvi = np.where(denom > 0, (nir - vis) / denom, 0)

    vegetation_mask = ndvi > 0.2
    veg_percentage = np.mean(vegetation_mask) * 100

    return {
        'ndvi_min': float(np.min(ndvi)),
        'ndvi_max': float(np.max(ndvi)),
        'ndvi_mean': float(np.mean(ndvi)),
        'ndvi_std': float(np.std(ndvi)),
        'vegetation_coverage': float(veg_percentage),
        'healthy_vegetation': float(np.mean(ndvi > 0.4) * 100),
    }


# ── Helpers ───────────────────────────────────────────────

def _curve_to_lut(points: list) -> np.ndarray:
    """Tonkurven-Punkte zu 256-Einträge LUT."""
    lut = np.zeros(256, dtype=np.uint8)
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
                    lut[i] = int(pts[j][1] + t * (pts[j + 1][1] - pts[j][1]))
                    break
    return lut


def ir_preset_to_xmp(preset: IRFalseColorPreset, filepath: str,
                       camera_model: str = "", filter_type: str = "720nm"):
    """Exportiert ein IR-Preset als Lightroom XMP."""
    import uuid

    sat_val = int((preset.saturation - 1.0) * 100)
    grayscale = "True" if preset.monochrome else "False"
    ir_filter = IR_FILTERS.get(filter_type, IR_FILTERS["720nm"])

    curve_items = ""
    if preset.tone_curve:
        items = "\n".join(f'    <rdf:li>{x}, {y}</rdf:li>' for x, y in preset.tone_curve)
        curve_items = f"""
   <crs:ToneCurvePV2012>
    <rdf:Seq>
{items}
    </rdf:Seq>
   </crs:ToneCurvePV2012>"""

    xmp = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - IR Preset">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:UUID="{str(uuid.uuid4()).upper()}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    crs:Copyright="DNG Channel Tool - IR Preset"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:Temperature="{ir_filter.typical_wb_temp + preset.wb_temp_shift}"
    crs:Tint="{ir_filter.typical_wb_tint + preset.wb_tint_shift}"
    crs:Contrast="{preset.contrast}"
    crs:Saturation="{sat_val}"
    crs:ConvertToGrayscale="{grayscale}"
    crs:Group{{Name}}="IR False Color"
    >{curve_items}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp)
