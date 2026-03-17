"""
Fujifilm Film Simulation Recipe Parser & Konverter

Parst Fujifilm-Rezepte (Text-Format wie auf fujixweekly.com)
und konvertiert sie zu:
- Adobe Lightroom XMP-Presets
- Nikon Picture Controls (NPC/NP3)
- ImageStyle für die Stil-Übertragung

Unterstützt alle 20+ Filmsimulationen und alle Parameter.
"""

import re
import uuid
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ── Fujifilm Film Simulations → Adobe Camera Profile Mapping ──

FILM_SIM_TO_ADOBE = {
    "PROVIA/STANDARD": "Camera PROVIA/STANDARD",
    "PROVIA": "Camera PROVIA/STANDARD",
    "Velvia/VIVID": "Camera Velvia/VIVID",
    "VELVIA": "Camera Velvia/VIVID",
    "ASTIA/SOFT": "Camera ASTIA/SOFT",
    "ASTIA": "Camera ASTIA/SOFT",
    "CLASSIC CHROME": "Camera CLASSIC CHROME",
    "CLASSIC CHROME": "Camera CLASSIC CHROME",
    "PRO Neg. Hi": "Camera PRO Neg. Hi",
    "PRO Neg. Std": "Camera PRO Neg. Std",
    "CLASSIC Neg.": "Camera CLASSIC Neg.",
    "CLASSIC NEG": "Camera CLASSIC Neg.",
    "ETERNA/CINEMA": "Camera ETERNA/CINEMA",
    "ETERNA": "Camera ETERNA/CINEMA",
    "ETERNA BLEACH BYPASS": "Camera ETERNA BLEACH BYPASS",
    "NOSTALGIC Neg.": "Camera NOSTALGIC Neg.",
    "NOSTALGIC NEG": "Camera NOSTALGIC Neg.",
    "REALA ACE": "Camera REALA ACE",
    "ACROS": "Camera ACROS",
    "ACROS+Ye": "Camera ACROS+Ye FILTER",
    "ACROS+R": "Camera ACROS+R FILTER",
    "ACROS+G": "Camera ACROS+G FILTER",
    "ACROS + Ye": "Camera ACROS+Ye FILTER",
    "ACROS + R": "Camera ACROS+R FILTER",
    "ACROS + G": "Camera ACROS+G FILTER",
    "MONOCHROME": "Camera MONOCHROME",
    "MONOCHROME+Ye": "Camera MONOCHROME+Ye FILTER",
    "MONOCHROME+R": "Camera MONOCHROME+R FILTER",
    "MONOCHROME+G": "Camera MONOCHROME+G FILTER",
    "SEPIA": "Camera SEPIA",
}

# WB Mode → Lightroom WhiteBalance + approx Temperature
WB_TO_LIGHTROOM = {
    "Auto": ("As Shot", 0),
    "Auto (White Priority)": ("As Shot", 0),
    "Auto (Ambience Priority)": ("As Shot", 0),
    "Daylight": ("Daylight", 5500),
    "Shade": ("Shade", 7500),
    "Cloudy": ("Cloudy", 6500),
    "Fluorescent Light-1": ("Fluorescent", 6000),
    "Fluorescent Light-2": ("Fluorescent", 4200),
    "Fluorescent Light-3": ("Fluorescent", 4900),
    "Fluorescent-1": ("Fluorescent", 6000),
    "Fluorescent-2": ("Fluorescent", 4200),
    "Fluorescent-3": ("Fluorescent", 4900),
    "Incandescent": ("Tungsten", 3000),
    "Underwater": ("Custom", 5000),
}

MONO_SIMS = {"ACROS", "ACROS+Ye", "ACROS+R", "ACROS+G", "ACROS + Ye",
             "ACROS + R", "ACROS + G", "MONOCHROME", "MONOCHROME+Ye",
             "MONOCHROME+R", "MONOCHROME+G", "SEPIA"}


@dataclass
class FujiRecipe:
    """Ein geparses Fujifilm-Rezept."""
    name: str = ""
    film_simulation: str = "PROVIA/STANDARD"

    # Grain
    grain_roughness: str = "Off"    # Off, Weak, Strong
    grain_size: str = ""            # Small, Large

    # Color Chrome
    color_chrome_effect: str = "Off"     # Off, Weak, Strong
    color_chrome_fx_blue: str = "Off"    # Off, Weak, Strong

    # White Balance
    wb_mode: str = "Auto"
    wb_kelvin: int = 0              # 0 = use preset, else Kelvin value
    wb_shift_red: int = 0           # -9 to +9
    wb_shift_blue: int = 0          # -9 to +9

    # Dynamic Range
    dynamic_range: str = "DR100"    # DR100, DR200, DR400, Auto

    # Tone
    highlight: float = 0            # -2 to +4 (0.5 steps)
    shadow: float = 0               # -2 to +4
    color: int = 0                  # -4 to +4
    sharpness: int = 0              # -4 to +4
    noise_reduction: int = 0        # -4 to +4
    clarity: int = 0                # -5 to +5

    # Monochrome-specific
    mono_color_wc: int = 0          # -18 to +18 Warm/Cool
    mono_color_mg: int = 0          # -18 to +18 Magenta/Green

    # Suggested settings (not saved in camera)
    iso_max: int = 0
    exposure_comp: str = ""

    @property
    def is_monochrome(self) -> bool:
        return self.film_simulation.upper().replace(" ", "") in {
            s.upper().replace(" ", "") for s in MONO_SIMS}

    @property
    def adobe_profile(self) -> str:
        # Try exact match first
        for key, val in FILM_SIM_TO_ADOBE.items():
            if key.upper().replace(" ", "").replace(".", "") == \
               self.film_simulation.upper().replace(" ", "").replace(".", ""):
                return val
        return f"Camera {self.film_simulation}"


def parse_recipe(text: str, name: str = "") -> FujiRecipe:
    """
    Parst ein Fujifilm-Rezept aus Text.

    Akzeptiert das Format von fujixweekly.com:
        Film Simulation: Classic Chrome
        Grain Effect: Weak, Small
        ...
    """
    recipe = FujiRecipe(name=name)

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        key, _, value = line.partition(':')
        key = key.strip().lower()
        value = value.strip()

        if 'film sim' in key:
            recipe.film_simulation = value

        elif 'grain' in key:
            parts = [p.strip() for p in value.split(',')]
            recipe.grain_roughness = parts[0] if parts else "Off"
            recipe.grain_size = parts[1] if len(parts) > 1 else ""

        elif 'color chrome' in key and 'blue' in key:
            recipe.color_chrome_fx_blue = value

        elif 'color chrome' in key:
            recipe.color_chrome_effect = value

        elif 'white balance' in key or key == 'wb':
            _parse_wb(recipe, value)

        elif 'dynamic' in key or key == 'dr':
            recipe.dynamic_range = value.upper().replace(" ", "")
            if not recipe.dynamic_range.startswith("DR"):
                recipe.dynamic_range = f"DR{recipe.dynamic_range}"

        elif 'highlight' in key:
            recipe.highlight = _parse_float(value)

        elif 'shadow' in key:
            recipe.shadow = _parse_float(value)

        elif key in ('color', 'farbe'):
            recipe.color = _parse_int(value)

        elif 'sharp' in key:
            recipe.sharpness = _parse_int(value)

        elif 'noise' in key or 'high iso' in key or key == 'nr':
            recipe.noise_reduction = _parse_int(value)

        elif 'clarity' in key:
            recipe.clarity = _parse_int(value)

        elif 'monochromatic' in key or 'mono' in key.replace(' ', ''):
            _parse_mono_color(recipe, value)

        elif 'iso' in key:
            m = re.search(r'(\d+)', value.replace(',', ''))
            if m:
                recipe.iso_max = int(m.group(1))

        elif 'exposure' in key or 'exp' in key:
            recipe.exposure_comp = value

    return recipe


def _parse_wb(recipe: FujiRecipe, value: str):
    """Parse White Balance Zeile."""
    # Format: "Daylight, +3 Red & -5 Blue" or "6700K, -1 Red & -6 Blue"
    parts = value.split(',', 1)
    mode_str = parts[0].strip()

    # Kelvin?
    kelvin_match = re.match(r'(\d+)\s*[kK]', mode_str)
    if kelvin_match:
        recipe.wb_kelvin = int(kelvin_match.group(1))
        recipe.wb_mode = f"{recipe.wb_kelvin}K"
    else:
        recipe.wb_mode = mode_str

    # Shift?
    if len(parts) > 1:
        shift_str = parts[1].strip()
        red_match = re.search(r'([+-]?\d+)\s*[Rr]', shift_str)
        blue_match = re.search(r'([+-]?\d+)\s*[Bb]', shift_str)
        if red_match:
            recipe.wb_shift_red = int(red_match.group(1))
        if blue_match:
            recipe.wb_shift_blue = int(blue_match.group(1))


def _parse_mono_color(recipe: FujiRecipe, value: str):
    """Parse Monochromatic Color: WC +3 & MG -2."""
    wc_match = re.search(r'WC\s*([+-]?\d+)', value, re.IGNORECASE)
    mg_match = re.search(r'MG\s*([+-]?\d+)', value, re.IGNORECASE)
    if wc_match:
        recipe.mono_color_wc = int(wc_match.group(1))
    if mg_match:
        recipe.mono_color_mg = int(mg_match.group(1))


def _parse_float(s: str) -> float:
    s = s.strip().replace(' ', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_int(s: str) -> int:
    s = s.strip().replace(' ', '')
    try:
        return int(float(s))
    except ValueError:
        return 0


# ── Export ────────────────────────────────────────────────

def recipe_to_xmp(recipe: FujiRecipe, filepath: str):
    """Konvertiert ein Fujifilm-Rezept zu einem Lightroom XMP-Preset."""
    preset_uuid = str(uuid.uuid4()).upper()
    preset_name = recipe.name or recipe.film_simulation

    # Camera profile
    profile = recipe.adobe_profile

    # White Balance
    lr_wb = "As Shot"
    lr_temp = 0
    if recipe.wb_kelvin > 0:
        lr_wb = "Custom"
        lr_temp = recipe.wb_kelvin
    else:
        for key, (lr_name, temp) in WB_TO_LIGHTROOM.items():
            if key.upper().replace(" ", "") == recipe.wb_mode.upper().replace(" ", ""):
                lr_wb = lr_name
                lr_temp = temp
                break

    # WB shift → Lightroom Tint (approximate)
    # Fuji Red/Blue shift → LR Temperature/Tint
    # Red+ = warmer (higher temp), Blue+ = cooler
    temp_shift = recipe.wb_shift_red * 150  # ~150K per step
    tint_shift = -recipe.wb_shift_blue * 3  # Approximate tint mapping

    if lr_temp > 0:
        lr_temp += temp_shift

    # Highlight/Shadow → Lightroom Highlights/Shadows
    # Fuji: -2 to +4, LR: -100 to +100
    lr_highlights = int(recipe.highlight * 25)  # Scale to -50..+100
    lr_shadows = int(-recipe.shadow * 25)       # Inverted: Fuji shadow+ = darker

    # Color → Saturation (-4..+4 → -100..+100)
    lr_saturation = int(recipe.color * 25)

    # Sharpness (-4..+4 → 0..150)
    lr_sharpness = max(0, 40 + recipe.sharpness * 15)

    # Clarity (-5..+5 → -100..+100)
    lr_clarity = int(recipe.clarity * 20)

    # NR (-4..+4 → 0..100)
    lr_luminance_nr = max(0, int(25 + recipe.noise_reduction * 8))

    # Grain
    grain_amount = 0
    grain_size = 25
    if recipe.grain_roughness.lower() == "weak":
        grain_amount = 25
    elif recipe.grain_roughness.lower() == "strong":
        grain_amount = 50
    if recipe.grain_size.lower() == "large":
        grain_size = 50

    # Monochrome
    grayscale = "True" if recipe.is_monochrome else "False"

    # Build temperature line
    temp_line = ""
    if lr_temp > 0:
        temp_line = f'    crs:Temperature="{lr_temp}"\n    crs:Tint="{tint_shift}"'
    else:
        temp_line = f'    crs:Tint="{tint_shift}"'

    xmp = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - Fuji Recipe">
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
    crs:Copyright="DNG Channel Tool - Fuji Recipe Converter"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:CameraProfile="{profile}"
    crs:WhiteBalance="{lr_wb}"
{temp_line}
    crs:Highlights2012="{lr_highlights}"
    crs:Shadows2012="{lr_shadows}"
    crs:Contrast="{int(recipe.highlight * 10 - recipe.shadow * 5)}"
    crs:Saturation="{lr_saturation}"
    crs:Sharpness="{lr_sharpness}"
    crs:Clarity="{lr_clarity}"
    crs:LuminanceSmoothing="{lr_luminance_nr}"
    crs:GrainAmount="{grain_amount}"
    crs:GrainSize="{grain_size}"
    crs:GrainFrequency="50"
    crs:ConvertToGrayscale="{grayscale}"
    crs:Group{{Name}}="Fujifilm Rezepte"
    />
 </rdf:RDF>
</x:xmpmeta>
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp)


def recipe_to_nikon_pc(recipe: FujiRecipe):
    """Konvertiert ein Fujifilm-Rezept zu Nikon Picture Control Parametern."""
    from npc_io import NikonPictureControlFile

    pc = NikonPictureControlFile()
    pc.name = (recipe.name or recipe.film_simulation)[:19]

    # Base profile mapping
    if recipe.is_monochrome:
        pc.base = "MONOCHROME"
    elif recipe.film_simulation.upper() in ("VELVIA/VIVID", "VELVIA"):
        pc.base = "VIVID"
    elif recipe.film_simulation.upper() in ("ASTIA/SOFT", "ASTIA"):
        pc.base = "PORTRAIT"
    elif recipe.film_simulation.upper() in ("CLASSIC CHROME", "CLASSIC NEG.", "CLASSIC NEG"):
        pc.base = "FLAT"
    elif recipe.film_simulation.upper() in ("ETERNA/CINEMA", "ETERNA"):
        pc.base = "FLAT"
    elif "BLEACH" in recipe.film_simulation.upper():
        pc.base = "STANDARD"
    elif "NOSTALGIC" in recipe.film_simulation.upper():
        pc.base = "STANDARD"
    else:
        pc.base = "STANDARD"

    # Parameters: Fuji range → Nikon unsigned (center 128)
    # Fuji Highlight/Shadow (-2..+4) → Nikon Contrast
    combined_contrast = (recipe.highlight - recipe.shadow) * 8
    pc.contrast = max(0, min(255, int(128 + combined_contrast)))

    # Brightness from shadow
    pc.brightness = max(0, min(255, int(128 - recipe.shadow * 10)))

    # Color → Saturation
    pc.saturation = max(0, min(255, int(128 + recipe.color * 16)))

    # Sharpness
    pc.sharpening = max(0, min(255, int(128 + recipe.sharpness * 16)))

    # Clarity
    pc.clarity = max(0, min(255, int(128 + recipe.clarity * 12)))

    return pc


def install_recipe_to_lightroom(recipe: FujiRecipe, subfolder: str = "Fujifilm Rezepte") -> str:
    """Installiert ein Fuji-Rezept direkt als Lightroom-Preset."""
    preset_dir = os.path.join(
        os.environ.get('APPDATA', ''),
        'Adobe', 'CameraRaw', 'Settings', subfolder)
    os.makedirs(preset_dir, exist_ok=True)

    name = recipe.name or recipe.film_simulation
    xmp_path = os.path.join(preset_dir, f"{name}.xmp")
    recipe_to_xmp(recipe, xmp_path)
    return xmp_path


# ── Example recipes for testing ──

EXAMPLE_RECIPES = {
    "Kodachrome 64": """Film Simulation: Classic Chrome
Grain Effect: Weak, Small
Color Chrome Effect: Strong
Color Chrome FX Blue: Off
White Balance: Daylight, +2 Red & -5 Blue
Dynamic Range: DR200
Highlight: 0
Shadow: +0.5
Color: +2
Sharpness: +1
High ISO NR: -4
Clarity: +3""",

    "Portra 400": """Film Simulation: Nostalgic Neg.
Grain Effect: Weak, Small
Color Chrome Effect: Weak
Color Chrome FX Blue: Weak
White Balance: Auto
Dynamic Range: DR200
Highlight: -1
Shadow: -1
Color: +1
Sharpness: -2
High ISO NR: -4
Clarity: 0""",

    "Tri-X 400 (S/W)": """Film Simulation: ACROS
Grain Effect: Strong, Small
Color Chrome Effect: Off
Color Chrome FX Blue: Off
White Balance: Daylight, 0 Red & 0 Blue
Dynamic Range: DR400
Highlight: +1
Shadow: -2
Sharpness: -1
High ISO NR: -4
Clarity: +2
Monochromatic Color: WC 0 & MG 0""",

    "CineStill 800T": """Film Simulation: Eterna/Cinema
Grain Effect: Weak, Large
Color Chrome Effect: Strong
Color Chrome FX Blue: Strong
White Balance: Incandescent, +4 Red & -2 Blue
Dynamic Range: DR200
Highlight: -1
Shadow: +1
Color: +3
Sharpness: -3
High ISO NR: -4
Clarity: -2""",

    "Classic Neg. Vintage": """Film Simulation: Classic Neg.
Grain Effect: Weak, Small
Color Chrome Effect: Strong
Color Chrome FX Blue: Weak
White Balance: Daylight, +3 Red & -3 Blue
Dynamic Range: DR200
Highlight: -0.5
Shadow: +1
Color: +2
Sharpness: -2
High ISO NR: -4
Clarity: -3""",
}
