"""
Nikon Picture Control (.NPC / .NP3) Reader/Writer

Formate:
- NCP v0100: Älteres Format (D-SLR Kameras, z.B. D850, D7500)
- NP3 v0300+: Neueres Format (Z-Serie Kameras, z.B. Z6 II, Z7 II, Z8, Z9)

Beide starten mit Magic "NCP\x00".
Die Tonkurve nutzt das gleiche Format wie NEFCurve1 in NEF-Dateien.

Dateien werden auf SD-Karte nach /NIKON/CUSTOMPC/ kopiert
und über Aufnahme-Menü → Bildstile verwalten → Laden/Speichern importiert.
"""

import struct
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Base profile IDs
BASE_PROFILES = {
    0: "STANDARD",
    1: "NEUTRAL",
    2: "VIVID",
    3: "MONOCHROME",
    4: "PORTRAIT",
    5: "LANDSCAPE",
    6: "FLAT",
    7: "DREAM",
    8: "MORNING",
    9: "POP",
    10: "SUNDAY",
    11: "SOMBER",
    12: "DRAMATIC",
    13: "SILENCE",
    14: "BLEACHED",
    15: "MELANCHOLIC",
    16: "PURE",
    17: "DENIM",
    18: "TOY",
    19: "SEPIA",
    20: "BLUE",
    21: "RED",
    22: "PINK",
    23: "CHARCOAL",
    24: "GRAPHITE",
    25: "BINARY",
    26: "CARBON",
    27: "RICH TONE",
}

BASE_PROFILE_BY_NAME = {v: k for k, v in BASE_PROFILES.items()}

FILTER_EFFECTS = {0: None, 1: "Yellow", 2: "Orange", 3: "Red", 4: "Green",
                  5: "Yellow-Green", 6: "Green"}
FILTER_BY_NAME = {v: k for k, v in FILTER_EFFECTS.items() if v}

TONING_EFFECTS = {0: None, 1: "B&W", 2: "Sepia", 3: "Cyanotype", 4: "Red",
                  5: "Yellow", 6: "Green", 7: "Blue-Green", 8: "Blue",
                  9: "Purple-Blue", 10: "Red-Purple"}


@dataclass
class NikonPictureControlFile:
    """Repräsentiert eine Nikon Picture Control Datei (.NPC/.NP3)."""
    name: str = "Custom"
    base: str = "STANDARD"
    version: str = "0100"

    # Parameters (0-255 range, center=128, None=Auto/0xFF)
    sharpening: Optional[int] = None    # Auto
    mid_range_sharpening: Optional[int] = None
    clarity: Optional[int] = None
    contrast: Optional[int] = 128       # 0 (neutral)
    brightness: Optional[int] = 128     # 0
    saturation: Optional[int] = 128     # 0
    hue: Optional[int] = 128            # 0

    # Monochrome options
    filter_effect: int = 0              # 0=Off, 1=Yellow, 2=Orange, 3=Red, 4=Green
    toning_effect: int = 0              # 0=Off, 1=B&W, 2=Sepia, ...
    toning_saturation: int = 0

    # Tone curve (list of (input, output) pairs, 0-255 range)
    tone_curve: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 0), (255, 255)])

    @property
    def is_monochrome(self) -> bool:
        return 'MONO' in self.base.upper() or self.base.upper() in (
            'SEPIA', 'BLUE', 'RED', 'PINK', 'CHARCOAL', 'GRAPHITE',
            'BINARY', 'CARBON')

    @property
    def base_id(self) -> int:
        return BASE_PROFILE_BY_NAME.get(self.base.upper(), 0)

    def get_param_display(self, val: Optional[int]) -> str:
        if val is None:
            return "Auto"
        return f"{val - 128:+d}"


def read_npc(filepath: str) -> NikonPictureControlFile:
    """Liest eine NPC/NP3-Datei."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if data[:3] != b'NCP':
        raise ValueError("Keine gültige NPC-Datei (Magic 'NCP' fehlt)")

    pc = NikonPictureControlFile()

    # Detect format version
    format_flags = data[4:8]
    version_str = data[12:16].decode('ascii', errors='replace')
    pc.version = version_str

    if format_flags == b'\x00\x00\x00\x01':
        # NCP v0100 format
        _read_ncp_v0100(data, pc)
    elif format_flags[2] == 0x01:
        # NP3 v0300+ format
        _read_np3_v0300(data, pc)
    else:
        # Try v0100 as fallback
        _read_ncp_v0100(data, pc)

    return pc


def _read_ncp_v0100(data: bytes, pc: NikonPictureControlFile):
    """Parse NCP v0100 format."""
    pc.name = data[16:36].decode('ascii', errors='replace').rstrip('\x00').strip()

    if len(data) > 45:
        params = data[36:]
        pc.base = BASE_PROFILES.get(params[0], "STANDARD")
        pc.filter_effect = params[2] if len(params) > 2 else 0
        pc.sharpening = None if params[3] == 0xFF else params[3]
        pc.contrast = params[4] if len(params) > 4 else 128
        pc.brightness = params[7] if len(params) > 7 else 128
        pc.saturation = params[8] if len(params) > 8 else 128
        pc.hue = None if (len(params) <= 9 or params[9] == 0xFF) else params[9]

    # Find tone curve (starts with "I0" marker)
    curve_off = data.find(b'I0')
    if curve_off >= 0 and curve_off + 10 < len(data):
        num_points = data[curve_off + 8]
        points = []
        for i in range(num_points):
            idx = curve_off + 9 + i * 2
            if idx + 1 < len(data):
                points.append((data[idx], data[idx + 1]))
        if points:
            pc.tone_curve = points


def _read_np3_v0300(data: bytes, pc: NikonPictureControlFile):
    """Parse NP3 v0300+ format."""
    pc.name = data[24:44].decode('ascii', errors='replace').rstrip('\x00').strip()

    # Base profile from section header at offset 46
    if len(data) > 46:
        pc.base = BASE_PROFILES.get(data[46], "STANDARD")

    # Filter/toning from offset 52-53
    if len(data) > 53:
        pc.filter_effect = data[52]
        pc.toning_effect = data[53] if data[53] != 0x4d else 0

    # Tagged entries: search for the parameter pattern
    # Parameters are encoded at specific offsets with value byte + flag byte
    # We identify them by the "00 02" marker pattern
    _parse_np3_params(data, pc)

    # Find tone curve
    curve_off = data.find(b'I0')
    if curve_off >= 0 and curve_off + 10 < len(data):
        num_points = data[curve_off + 8]
        points = []
        for i in range(num_points):
            idx = curve_off + 9 + i * 2
            if idx + 1 < len(data):
                points.append((data[idx], data[idx + 1]))
        if points:
            pc.tone_curve = points


def _parse_np3_params(data: bytes, pc: NikonPictureControlFile):
    """Extract parameters from NP3 tagged entries."""
    # Look for the 00 02 marker pattern and extract value + flag pairs
    values = []
    for i in range(44, min(len(data) - 2, 300)):
        if data[i] == 0x00 and data[i + 1] == 0x02 and i + 3 < len(data):
            val = data[i + 2]
            flag = data[i + 3]
            values.append((val, flag))

    # Map extracted values to parameters
    # Based on analysis: values appear in order for sharpening, contrast, etc.
    if len(values) >= 7:
        # First few 00 02 patterns correspond to picture control params
        auto_or_val = lambda v, f: None if v == 0xFF else v
        idx = 0
        # Skip base-related entries (first 1-2)
        if len(values) > 10:
            # Typical order: base_params, sharpening, mid_sharp, clarity,
            # contrast, brightness, saturation, hue, ...
            pc.sharpening = auto_or_val(values[1][0], values[1][1]) if len(values) > 1 else None
            pc.contrast = values[2][0] if len(values) > 2 and values[2][0] != 0xFF else 128
            pc.brightness = values[3][0] if len(values) > 3 and values[3][0] != 0xFF else 128
            pc.saturation = values[4][0] if len(values) > 4 and values[4][0] != 0xFF else 128
            pc.hue = auto_or_val(values[5][0], values[5][1]) if len(values) > 5 else None


def write_npc(filepath: str, pc: NikonPictureControlFile, format_version: str = "0100"):
    """
    Schreibt eine Nikon Picture Control Datei.

    Args:
        filepath: Ausgabepfad (.NPC für v0100, .NP3 für v0300)
        pc: Picture Control Daten
        format_version: "0100" für NCP (D-SLR) oder "0300" für NP3 (Z-Serie)
    """
    if format_version.startswith("03"):
        _write_np3(filepath, pc)
    else:
        _write_ncp_v0100(filepath, pc)


def _write_ncp_v0100(filepath: str, pc: NikonPictureControlFile):
    """Write NCP v0100 format (compatible with most Nikon DSLRs)."""
    out = bytearray()

    # Header
    out += b'NCP\x00'                   # Magic
    out += b'\x00\x00\x00\x01'          # Format version 1
    out += b'\x00\x00\x00\x24'          # Data offset = 36
    out += b'0100'                       # Version string

    # Name (20 bytes, null-padded)
    name_bytes = pc.name[:19].encode('ascii', errors='replace')
    out += name_bytes + b'\x00' * (20 - len(name_bytes))

    # Parameters (fixed layout)
    out += bytes([pc.base_id])           # Base profile ID
    out += b'\xc2'                       # Unknown flag
    out += bytes([pc.filter_effect])     # Filter effect
    out += bytes([0xFF if pc.sharpening is None else pc.sharpening])
    out += bytes([pc.contrast or 128])
    out += b'\x01\x01'                   # Unknown
    out += bytes([pc.brightness or 128])
    out += bytes([pc.saturation or 128])
    out += bytes([0xFF if pc.hue is None else pc.hue])
    out += b'\xff\xff'                   # Unknown (auto values)
    out += b'\x00\x00'                   # Padding

    # Tone curve section marker
    out += b'\x00\x02'                   # Section count/type
    out += b'\x00\x00'                   # Padding
    out += struct.pack('>H', 0x0242)     # Size marker

    # Tone curve data (NEFCurve1 format)
    out += _encode_tone_curve(pc.tone_curve)

    with open(filepath, 'wb') as f:
        f.write(out)


def _write_np3(filepath: str, pc: NikonPictureControlFile):
    """Write NP3 v0300 format (for Z-series cameras)."""
    out = bytearray()

    # Header
    out += b'NCP\x00'                    # Magic
    out += b'\x00\x00\x01\x00'           # Format NP3
    out += b'\x00\x00\x00\x04'           # Type flag
    out += b'0300'                        # Version

    # Sub-header
    out += b'\x00\x00\x02\x00'           # Flags
    out += b'\x00\x00\x00\x14'           # Name length (20)

    # Name (20 bytes)
    name_bytes = pc.name[:19].encode('ascii', errors='replace')
    out += name_bytes + b'\x00' * (20 - len(name_bytes))

    # Padding
    out += b'\x00\x00'

    # Section header with base info
    out += bytes([pc.base_id])           # Base ID
    out += b'\x00\x00\x00\x00\x02'
    out += bytes([pc.filter_effect])
    out += bytes([0x4d if pc.is_monochrome else 0x00])

    # Tagged parameter entries (10 bytes each)
    def _entry(tag: int, value: int, flag: int = 0x04):
        return bytes([tag, 0x00, 0x00, 0x00, 0x00, 0x02,
                      value, flag, 0x00, 0x00])

    auto = lambda v: 0xFF if v is None else v

    out += _entry(0x04, 0x00, 0x00)                    # Adjust mode
    out += _entry(0x05, auto(pc.sharpening), 0x01)     # Sharpening
    out += _entry(0x06, auto(pc.mid_range_sharpening), 0x04)  # Mid-range sharpening
    out += _entry(0x07, auto(pc.clarity), 0x04)        # Clarity
    out += _entry(0x08, pc.contrast or 128, 0x04)      # Contrast
    out += _entry(0x09, pc.brightness or 128, 0x04)    # Brightness
    out += _entry(0x0A, auto(pc.saturation), 0x04)     # Saturation
    out += _entry(0x0B, auto(pc.hue), 0x04)            # Hue
    out += _entry(0x0C, 128, 0x00)                     # Reserved
    out += _entry(0x0D, 128, 0x00)                     # Reserved
    out += _entry(0x0E, 0xFF, 0x04)                    # Auto param
    out += _entry(0x0F, 0xFF, 0x01)
    out += _entry(0x10, 0xFF, 0x01)
    out += _entry(0x11, 0xFF, 0x01)
    out += _entry(0x12, 0xFF, 0x01)
    out += _entry(0x13, 0xFF, 0x01)
    out += _entry(0x14, 128, 0x01)                     # Toning sat
    out += _entry(0x15, 0xFF, 0x0A)                    # Auto
    out += _entry(0x16, pc.toning_saturation or 128, 0x04)

    # End marker + tone curve flag
    has_curve = len(pc.tone_curve) > 2
    out += bytes([0x00, 0x01])
    out += bytes([0x01 if has_curve else 0x00])
    out += b'\x00\x00\x00\x00'

    # Tone curve
    if has_curve:
        # Padding to align
        out += b'\x00' * (4 - len(out) % 4) if len(out) % 4 else b''
        out += b'\x00\x00\x00\x02\x00\x00\x02\x42'  # Curve section header
        out += _encode_tone_curve(pc.tone_curve)

    with open(filepath, 'wb') as f:
        f.write(out)


def _encode_tone_curve(points: List[Tuple[int, int]]) -> bytes:
    """Encode tone curve in Nikon NEFCurve1 format."""
    out = bytearray()

    # Header
    out += b'I0'                         # Curve version "I0"
    out += b'\x00\xff'                   # Input range 0-255
    out += b'\x00\xff'                   # Output range 0-255
    out += b'\x01\x0f'                   # Flags

    # Number of control points
    num = min(len(points), 20)
    out += bytes([num])

    # Control points (input, output pairs)
    for i in range(num):
        x, y = points[i]
        out += bytes([max(0, min(255, x)), max(0, min(255, y))])

    # Pad remaining point slots (up to 20 points)
    for _ in range(num, 20):
        out += b'\x00\x00'

    # Lookup table (256 entries, 16-bit big-endian)
    # Interpolate from control points
    lut = _interpolate_curve(points, 256)
    for val in lut:
        out += struct.pack('>H', max(0, min(4095, int(val * 16))))

    return bytes(out)


def _interpolate_curve(points: List[Tuple[int, int]], num_output: int = 256) -> List[float]:
    """Interpolate tone curve from control points using linear interpolation."""
    if not points:
        return list(range(num_output))

    # Sort by input value
    pts = sorted(points, key=lambda p: p[0])

    result = []
    for i in range(num_output):
        x = i

        # Find surrounding points
        if x <= pts[0][0]:
            result.append(pts[0][1])
        elif x >= pts[-1][0]:
            result.append(pts[-1][1])
        else:
            # Linear interpolation between surrounding points
            for j in range(len(pts) - 1):
                if pts[j][0] <= x <= pts[j + 1][0]:
                    t = (x - pts[j][0]) / max(1, pts[j + 1][0] - pts[j][0])
                    y = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
                    result.append(y)
                    break

    return result


# ── SD-Card Installation ──────────────────────────────────

def find_sd_cards() -> List[str]:
    """Findet SD-Karten/Wechseldatenträger unter Windows."""
    import string
    cards = []
    for letter in string.ascii_uppercase[3:]:  # D: onwards
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            # Check if it looks like a camera card
            nikon_dir = os.path.join(drive, "NIKON")
            dcim_dir = os.path.join(drive, "DCIM")
            if os.path.isdir(nikon_dir) or os.path.isdir(dcim_dir):
                cards.append(drive)
            elif os.path.isdir(drive):
                # Any removable drive
                try:
                    import ctypes
                    drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
                    if drive_type == 2:  # DRIVE_REMOVABLE
                        cards.append(drive)
                except Exception:
                    pass
    return cards


def install_to_camera(npc_path: str, sd_card: str) -> str:
    """
    Kopiert eine NPC/NP3-Datei auf die SD-Karte für den Kamera-Import.

    Erstellt /NIKON/CUSTOMPC/ auf der Karte falls nötig.
    Gibt den Zielpfad zurück.
    """
    dest_dir = os.path.join(sd_card, "NIKON", "CUSTOMPC")
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(npc_path).upper()
    # NPC filenames must be PICCON01-PICCON99 for older cameras
    # or any name for newer cameras

    dest = os.path.join(dest_dir, filename)
    shutil.copy2(npc_path, dest)
    return dest


# ── Conversion Helpers ────────────────────────────────────

def from_nef_picture_control(nef_pc) -> NikonPictureControlFile:
    """
    Konvertiert einen extrahierten NEF PictureControl
    (aus nef_extract.NikonPictureControl) zu einer NPC-Datei.
    """
    pc = NikonPictureControlFile()
    pc.name = nef_pc.name or "Custom"
    pc.base = nef_pc.base or "STANDARD"

    # Convert from signed (-128..+127) to unsigned (0..255)
    def to_unsigned(val):
        if val is None:
            return None
        return max(0, min(255, val + 128))

    pc.sharpening = to_unsigned(nef_pc.sharpening)
    pc.clarity = to_unsigned(nef_pc.clarity)
    pc.contrast = to_unsigned(nef_pc.contrast)
    pc.brightness = to_unsigned(nef_pc.brightness)
    pc.saturation = to_unsigned(nef_pc.saturation)
    pc.hue = to_unsigned(nef_pc.hue)

    if nef_pc.tone_curve:
        pc.tone_curve = list(nef_pc.tone_curve)

    return pc


def to_lightroom_values(pc: NikonPictureControlFile) -> dict:
    """Konvertiert NPC-Werte zu Lightroom-Parametern."""
    def to_lr(val, lr_max=100):
        if val is None:
            return 0
        return int((val - 128) / 128 * lr_max)

    return {
        'contrast': to_lr(pc.contrast),
        'brightness': to_lr(pc.brightness),
        'saturation': to_lr(pc.saturation),
        'sharpness': max(0, to_lr(pc.sharpening, 150)) if pc.sharpening else 40,
        'clarity': to_lr(pc.clarity),
        'is_monochrome': pc.is_monochrome,
        'tone_curve': pc.tone_curve,
    }
