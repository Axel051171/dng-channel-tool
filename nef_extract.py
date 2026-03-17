"""
NEF Picture Control Extractor

Extrahiert Nikon Picture Controls, Tonkurven und Farbeinstellungen
aus NEF-Dateien und konvertiert sie zu Adobe Lightroom XMP-Presets.

Unterstützt:
- Picture Control Name & Basis (Standard, Vivid, Monochrome, Flat, etc.)
- Tonkurve (NEFCurve1) mit beliebig vielen Kontrollpunkten
- Weißabgleich (Modus + R/B-Koeffizienten)
- Active D-Lighting, Farbraum, Vignettenkorrektur
- Kontrast, Helligkeit, Sättigung, Farbton, Schärfe, Klarheit
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import exifread
except ImportError:
    exifread = None

try:
    import rawpy
except ImportError:
    rawpy = None


@dataclass
class NikonPictureControl:
    """Extrahierte Nikon Picture Control Daten."""
    name: str = ""
    base: str = ""
    version: str = ""

    # Tone curve points (0-255 range)
    tone_curve: List[Tuple[int, int]] = field(default_factory=list)

    # Picture Control parameters (-100 to 100 range, None = Auto)
    sharpening: Optional[float] = None
    mid_range_sharpening: Optional[float] = None
    clarity: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    saturation: Optional[float] = None
    hue: Optional[float] = None

    # Filter effects (Monochrome)
    filter_effect: Optional[str] = None  # Yellow, Orange, Red, Green
    toning_effect: Optional[str] = None  # Sepia, Blue, etc.
    toning_saturation: Optional[int] = None

    # White Balance
    wb_mode: str = ""
    wb_r_coeff: float = 1.0
    wb_b_coeff: float = 1.0
    color_temp_auto: int = 0

    # Other settings
    color_space: str = ""
    active_d_lighting: str = ""
    vignette_control: str = ""
    high_iso_nr: str = ""
    is_monochrome: bool = False

    # Preview image data
    preview_data: Optional[bytes] = None


# Nikon Filter Effect codes
FILTER_EFFECTS = {
    0: None,
    1: "Yellow",
    2: "Orange",
    3: "Red",
    4: "Green",
}

# Nikon Toning Effect codes
TONING_EFFECTS = {
    0: None,
    1: "B&W",
    2: "Sepia",
    3: "Cyanotype",
    4: "Red",
    5: "Yellow",
    6: "Green",
    7: "Blue-Green",
    8: "Blue",
    9: "Purple-Blue",
    10: "Red-Purple",
}

# Nikon White Balance modes to Lightroom
WB_MODE_MAP = {
    "AUTO": "As Shot",
    "AUTO1": "As Shot",
    "AUTO2": "As Shot",
    "SUNNY": "Daylight",
    "CLOUDY": "Cloudy",
    "SHADE": "Shade",
    "INCANDESCENT": "Tungsten",
    "FLUORESCENT": "Fluorescent",
    "FLASH": "Flash",
    "MANUAL": "Custom",
}


def extract_picture_control(nef_path: str) -> NikonPictureControl:
    """
    Extrahiert Picture Control Daten aus einer Nikon NEF-Datei.
    """
    if exifread is None:
        raise ImportError("exifread ist nicht installiert: pip install exifread")

    pc = NikonPictureControl()

    with open(nef_path, 'rb') as f:
        tags = exifread.process_file(f, details=True)

    # ── Picture Control Block ──
    pc_tag = tags.get('MakerNote PictureControl')
    if pc_tag:
        vals = pc_tag.values
        pc.version = bytes(vals[:8]).decode('ascii', errors='replace')
        pc.name = bytes(vals[8:28]).decode('ascii', errors='replace').rstrip('\x00').strip()
        pc.base = bytes(vals[28:48]).decode('ascii', errors='replace').rstrip('\x00').strip()
        pc.is_monochrome = 'MONO' in pc.base.upper()

        # Picture Control v0300 parameters
        if len(vals) > 75 and pc.version.startswith('0300'):
            _parse_pc_v0300(pc, vals)

    # ── Tone Curve ──
    curve_tag = tags.get('MakerNote NEFCurve1')
    if curve_tag:
        vals = curve_tag.values
        if len(vals) > 10:
            num_points = vals[8]
            points = []
            for i in range(num_points):
                idx = 9 + i * 2
                if idx + 1 < len(vals):
                    x = vals[idx]
                    y = vals[idx + 1]
                    points.append((x, y))
            pc.tone_curve = points

    # ── White Balance ──
    wb_tag = tags.get('MakerNote Whitebalance')
    if wb_tag:
        pc.wb_mode = str(wb_tag).strip()

    wb_coeff = tags.get('MakerNote WhiteBalanceRBCoeff')
    if wb_coeff:
        try:
            ratios = str(wb_coeff).split(', ')
            if len(ratios) >= 2:
                # Sicheres Parsen von Ratio-Strings wie "256/128"
                def _parse_ratio(s: str) -> float:
                    s = s.strip()
                    if '/' in s:
                        num, den = s.split('/', 1)
                        return float(num) / float(den)
                    return float(s)
                pc.wb_r_coeff = _parse_ratio(ratios[0])
                pc.wb_b_coeff = _parse_ratio(ratios[1])
        except (ValueError, ZeroDivisionError):
            pass

    ct_auto = tags.get('MakerNote ColorTemperatureAuto')
    if ct_auto:
        try:
            pc.color_temp_auto = int(str(ct_auto))
        except ValueError:
            pass

    # ── Other settings ──
    cs = tags.get('MakerNote ColorSpace')
    if cs:
        pc.color_space = str(cs)

    adl = tags.get('MakerNote ActiveDLighting')
    if adl:
        pc.active_d_lighting = str(adl)

    vc = tags.get('MakerNote VignetteControl')
    if vc:
        pc.vignette_control = str(vc)

    hiso = tags.get('MakerNote HighISONoiseReduction')
    if hiso:
        pc.high_iso_nr = str(hiso)

    # ── Extract Preview ──
    if rawpy is not None:
        try:
            with rawpy.imread(nef_path) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    pc.preview_data = thumb.data
        except Exception:
            pass

    return pc


def _parse_pc_v0300(pc: NikonPictureControl, vals: list):
    """Parse Picture Control version 0300 parameters."""
    # The parameter block starts at offset 48
    # Format appears to use pairs where even bytes are flags and odd bytes are values
    # Based on analysis of real Nikon Z files:

    # Offset 49: Filter Effect (for mono)
    if vals[49] in FILTER_EFFECTS:
        pc.filter_effect = FILTER_EFFECTS[vals[49]]

    # Offset 50: Toning Effect
    if vals[50] in TONING_EFFECTS:
        pc.toning_effect = TONING_EFFECTS[vals[50]]

    # Offset 51: Toning Saturation
    pc.toning_saturation = vals[51]

    # Parameters stored with flag-byte pattern
    # Each parameter seems to be at (flag_offset, value_offset)
    # Value 0xFF = Auto, otherwise center is 128, range +-127
    def decode_param(flag_byte, val_byte):
        if val_byte == 0xFF:
            return None  # Auto
        # Signed value centered at 128
        return (val_byte - 128)

    # Based on byte pattern analysis:
    # Bytes 52-53: Sharpening
    # Bytes 54-55: Mid-Range Sharpening
    # Bytes 56-57: Clarity
    # Bytes 58-59: Contrast
    # Bytes 60-61: Brightness
    # Bytes 62-63: Saturation
    # Bytes 64-65: Hue

    if len(vals) > 65:
        pc.sharpening = decode_param(vals[52], vals[53])
        pc.mid_range_sharpening = decode_param(vals[54], vals[55])
        pc.clarity = decode_param(vals[56], vals[57])
        pc.contrast = decode_param(vals[58], vals[59])
        pc.brightness = decode_param(vals[60], vals[61])
        pc.saturation = decode_param(vals[62], vals[63])
        pc.hue = decode_param(vals[64], vals[65])


def picture_control_to_xmp(pc: NikonPictureControl, filepath: str,
                            preset_name: str = None,
                            camera_model: str = "NIKON Z 6_2"):
    """
    Konvertiert einen extrahierten Picture Control zu einem
    Adobe Lightroom XMP-Preset.
    """
    if preset_name is None:
        preset_name = pc.name or "Nikon Picture Control"

    preset_uuid = str(uuid.uuid4()).upper()

    # Map Nikon WB to Lightroom WB
    wb_mode_clean = pc.wb_mode.strip().upper().replace(' ', '')
    lr_wb = WB_MODE_MAP.get(wb_mode_clean, "As Shot")

    # Build tone curve string
    curve_points = []
    if pc.tone_curve:
        for x, y in pc.tone_curve:
            curve_points.append(f"    <rdf:li>{x}, {y}</rdf:li>")
    else:
        curve_points = [
            "    <rdf:li>0, 0</rdf:li>",
            "    <rdf:li>255, 255</rdf:li>",
        ]
    curve_xml = "\n".join(curve_points)

    # Map Nikon parameter values to Lightroom range
    # Nikon: centered at 0, range roughly -127 to +127
    # Lightroom: different ranges per parameter
    def nikon_to_lr(val, lr_range=100, nikon_max=50):
        """Scale Nikon value to Lightroom range."""
        if val is None:
            return 0
        return int(max(-lr_range, min(lr_range, val * lr_range / nikon_max)))

    lr_contrast = nikon_to_lr(pc.contrast, 100, 50)
    lr_brightness = nikon_to_lr(pc.brightness, 100, 50)
    lr_saturation = nikon_to_lr(pc.saturation, 100, 50)
    lr_sharpness = max(0, nikon_to_lr(pc.sharpening, 150, 50)) if pc.sharpening is not None else 40
    lr_clarity = nikon_to_lr(pc.clarity, 100, 50)

    # Monochrome settings
    is_mono = pc.is_monochrome
    grayscale = "True" if is_mono else "False"

    # Toning for monochrome (Sepia etc.)
    toning_hue = 0
    toning_sat = 0
    if is_mono and pc.toning_effect and pc.toning_effect != "B&W":
        toning_sat = pc.toning_saturation or 30
        toning_map = {
            "Sepia": 47, "Cyanotype": 215, "Red": 10,
            "Yellow": 60, "Green": 120, "Blue-Green": 175,
            "Blue": 230, "Purple-Blue": 260, "Red-Purple": 330,
        }
        toning_hue = toning_map.get(pc.toning_effect, 0)

    # Filter effect for mono (simulated via color mixer)
    filter_adjustments = ""
    if is_mono and pc.filter_effect:
        # Simulate color filter in B&W conversion
        filter_map = {
            "Yellow": {"Red": 30, "Orange": 50, "Yellow": 80, "Green": 10, "Blue": -50},
            "Orange": {"Red": 50, "Orange": 80, "Yellow": 40, "Green": -10, "Blue": -70},
            "Red": {"Red": 80, "Orange": 50, "Yellow": 10, "Green": -40, "Blue": -80},
            "Green": {"Red": -30, "Orange": -10, "Yellow": 20, "Green": 80, "Blue": -20},
        }
        adjustments = filter_map.get(pc.filter_effect, {})
        for color, val in adjustments.items():
            filter_adjustments += f'\n    crs:GrayMixer{color}="{val}"'

    xmp_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - NEF Extractor">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{preset_uuid}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    crs:CameraModelRestriction="{camera_model}"
    crs:Copyright="DNG Channel Tool"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:WhiteBalance="{lr_wb}"
    crs:Contrast="{lr_contrast}"
    crs:Saturation="{lr_saturation}"
    crs:Sharpness="{lr_sharpness}"
    crs:Clarity="{lr_clarity}"
    crs:ConvertToGrayscale="{grayscale}"
    crs:ToneCurveName2012="Custom"
    crs:SplitToningShadowHue="{toning_hue}"
    crs:SplitToningShadowSaturation="{toning_sat}"
    crs:SplitToningHighlightHue="{toning_hue}"
    crs:SplitToningHighlightSaturation="{toning_sat}"
    crs:SplitToningBalance="0"{filter_adjustments}
    crs:Group{{Name}}="Nikon Picture Controls"
    >
   <crs:ToneCurvePV2012>
    <rdf:Seq>
{curve_xml}
    </rdf:Seq>
   </crs:ToneCurvePV2012>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp_content)

    return filepath


def save_preview(pc: NikonPictureControl, filepath: str) -> Optional[str]:
    """Speichert das eingebettete Vorschaubild."""
    if pc.preview_data:
        with open(filepath, 'wb') as f:
            f.write(pc.preview_data)
        return filepath
    return None


def print_picture_control(pc: NikonPictureControl):
    """Gibt die Picture Control Infos formatiert aus."""
    print(f"  Name:       {pc.name}")
    print(f"  Basis:      {pc.base}")
    print(f"  Version:    {pc.version}")
    print(f"  Monochrom:  {'Ja' if pc.is_monochrome else 'Nein'}")
    print()

    if pc.tone_curve:
        print(f"  Tonkurve ({len(pc.tone_curve)} Punkte):")
        for x, y in pc.tone_curve:
            bar = '#' * (y * 30 // 255)
            print(f"    ({x:3d}, {y:3d}) |{bar}")
    print()

    params = [
        ("Schaerfe", pc.sharpening),
        ("MidRange-Schaerfe", pc.mid_range_sharpening),
        ("Klarheit", pc.clarity),
        ("Kontrast", pc.contrast),
        ("Helligkeit", pc.brightness),
        ("Saettigung", pc.saturation),
        ("Farbton", pc.hue),
    ]
    print("  Parameter:")
    for name, val in params:
        if val is None:
            print(f"    {name:20s}: Auto")
        else:
            print(f"    {name:20s}: {val:+d}")
    print()

    if pc.filter_effect:
        print(f"  Filtereffekt: {pc.filter_effect}")
    if pc.toning_effect:
        print(f"  Toning: {pc.toning_effect} (Saettigung: {pc.toning_saturation})")
    print()

    print(f"  Weissabgleich: {pc.wb_mode}")
    print(f"  WB R-Koeff:    {pc.wb_r_coeff:.4f}")
    print(f"  WB B-Koeff:    {pc.wb_b_coeff:.4f}")
    print(f"  Farbraum:      {pc.color_space}")
    print(f"  Active D-Lighting: {pc.active_d_lighting}")
    print(f"  Vignettierung: {pc.vignette_control}")
    print(f"  High-ISO NR:   {pc.high_iso_nr}")
    if pc.preview_data:
        print(f"  Vorschaubild:  {len(pc.preview_data)} Bytes")


# ── CLI ──
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Verwendung: python nef_extract.py <datei.nef> [ausgabe.xmp]")
        sys.exit(1)

    nef_path = sys.argv[1]
    xmp_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Lese: {nef_path}")
    print()

    pc = extract_picture_control(nef_path)

    print("=== Nikon Picture Control ===")
    print_picture_control(pc)

    if xmp_path is None:
        base = os.path.splitext(nef_path)[0]
        xmp_path = f"{base}_{pc.name or 'preset'}.xmp"

    picture_control_to_xmp(pc, xmp_path)
    print(f"\nLightroom-Preset gespeichert: {xmp_path}")

    # Save preview
    preview_path = os.path.splitext(nef_path)[0] + "_preview.jpg"
    if save_preview(pc, preview_path):
        print(f"Vorschaubild gespeichert: {preview_path}")
