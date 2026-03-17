"""
DNG Writer - Erzeugt DNG-Dateien aus PGM-Rohdaten

Konvertiert PGM (Portable GrayMap, P5 Binary) Sensordaten zu
DNG (Digital Negative) Dateien mit:
- Bayer-CFA-Pattern (RGGB, BGGR, GRBG, GBRG) oder Monochrom
- Farbmatrizen (ColorMatrix1/2, ForwardMatrix1/2)
- Weißabgleich (AsShotNeutral)
- Black/White-Level
- Eingebettetes Kameraprofil

Inspiriert von pgm2dng (fastvideo), komplett in Python auf Basis
der bestehenden TIFF/IFD-Infrastruktur aus dcp_io.py umgesetzt.
"""

import struct
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from dcp_io import (
    TIFF_BYTE, TIFF_ASCII, TIFF_SHORT, TIFF_LONG, TIFF_RATIONAL, TIFF_SRATIONAL,
    TIFF_TYPE_SIZE,
    _float_to_srational,
    TAG_UNIQUE_CAMERA_MODEL,
    TAG_COLOR_MATRIX_1, TAG_COLOR_MATRIX_2,
    TAG_CALIBRATION_ILLUMINANT_1, TAG_CALIBRATION_ILLUMINANT_2,
    TAG_FORWARD_MATRIX_1, TAG_FORWARD_MATRIX_2,
    TAG_PROFILE_NAME,
    ILLUMINANT_A, ILLUMINANT_D65,
)


# ── Standard TIFF Tags ───────────────────────────────────────

TAG_IMAGE_WIDTH = 256                   # 0x0100
TAG_IMAGE_LENGTH = 257                  # 0x0101
TAG_BITS_PER_SAMPLE = 258              # 0x0102
TAG_COMPRESSION = 259                   # 0x0103
TAG_PHOTOMETRIC_INTERPRETATION = 262    # 0x0106
TAG_STRIP_OFFSETS = 273                 # 0x0111
TAG_SAMPLES_PER_PIXEL = 277            # 0x0115
TAG_ROWS_PER_STRIP = 278               # 0x0116
TAG_STRIP_BYTE_COUNTS = 279            # 0x0117

# ── TIFF/EP CFA Tags ─────────────────────────────────────────

TAG_CFA_REPEAT_PATTERN_DIM = 33421     # 0x828D
TAG_CFA_PATTERN_TIFFEP = 33422         # 0x828E

# ── DNG-spezifische Tags ─────────────────────────────────────

TAG_DNG_VERSION = 50706                 # 0xC612
TAG_DNG_BACKWARD_VERSION = 50707        # 0xC613
TAG_CFA_PLANE_COLOR = 50710             # 0xC616
TAG_CFA_LAYOUT = 50711                  # 0xC617
TAG_BLACK_LEVEL = 50714                 # 0xC61A
TAG_WHITE_LEVEL = 50717                 # 0xC61D
TAG_DEFAULT_SCALE = 50718               # 0xC61E
TAG_DEFAULT_CROP_ORIGIN = 50719         # 0xC61F
TAG_DEFAULT_CROP_SIZE = 50720           # 0xC620
TAG_AS_SHOT_NEUTRAL = 50728             # 0xC628
TAG_BASELINE_EXPOSURE = 50730           # 0xC62A

# ── Photometric Interpretation Werte ─────────────────────────

PI_CFA = 32803          # Color Filter Array (Bayer)
PI_LINEAR_RAW = 34892   # Linear Raw (Monochrom)

# ── CFA Pattern Mapping ──────────────────────────────────────
# R=0, G=1, B=2

CFA_PATTERNS = {
    "RGGB": bytes([0, 1, 1, 2]),
    "BGGR": bytes([2, 1, 1, 0]),
    "GRBG": bytes([1, 0, 2, 1]),
    "GBRG": bytes([1, 2, 0, 1]),
}

TIFF_MAGIC = 42


# ── Datenstrukturen ──────────────────────────────────────────

@dataclass
class PGMData:
    """Geladene PGM-Datei."""
    width: int
    height: int
    max_val: int
    bits_per_sample: int    # 8 oder 16
    pixel_data: bytes       # Raw Pixeldaten (Little-Endian für 16-bit)


@dataclass
class DNGConfig:
    """Konfiguration für die DNG-Erzeugung."""
    # Kamera
    camera_model: str = "PGM Sensor"

    # CFA-Pattern: "RGGB", "BGGR", "GRBG", "GBRG" oder "MONO"
    cfa_pattern: str = "RGGB"

    # Pegel
    black_level: int = 0
    white_level: Optional[int] = None   # None = max_val aus PGM

    # Farbmatrizen (3x3 numpy, XYZ → Camera RGB)
    color_matrix_1: Optional[np.ndarray] = None
    color_matrix_2: Optional[np.ndarray] = None
    illuminant_1: int = ILLUMINANT_A
    illuminant_2: int = ILLUMINANT_D65

    # Forward-Matrizen (optional)
    forward_matrix_1: Optional[np.ndarray] = None
    forward_matrix_2: Optional[np.ndarray] = None

    # Weißabgleich: (R, G, B) Neutral-Werte, z.B. (0.47, 1.0, 0.63)
    as_shot_neutral: Optional[Tuple[float, float, float]] = None

    # Crop
    default_crop_origin: Tuple[int, int] = (0, 0)
    default_crop_size: Optional[Tuple[int, int]] = None  # None = (width, height)
    default_scale: Tuple[int, int] = (1, 1)

    # Profil
    profile_name: str = ""
    baseline_exposure: float = 0.0

    # DNG-Version
    dng_version: Tuple[int, int, int, int] = (1, 4, 0, 0)
    dng_backward_version: Tuple[int, int, int, int] = (1, 4, 0, 0)


# ── PGM Reader ───────────────────────────────────────────────

def read_pgm(filepath: str) -> PGMData:
    """
    Liest eine PGM-Datei (P5 Binary Format).

    Unterstützt 8-bit (maxval ≤ 255) und 16-bit (maxval > 255).
    16-bit PGM-Daten werden von Big-Endian zu Little-Endian konvertiert.
    """
    with open(filepath, 'rb') as f:
        # Magic
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Keine gültige PGM-Datei (erwartet P5, bekam {magic!r})")

        # Header-Felder lesen, Kommentare überspringen
        tokens = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError("Unerwartetes Dateiende im PGM-Header")
            line = line.strip()
            if line.startswith(b'#'):
                continue
            tokens.extend(line.split())

        width = int(tokens[0])
        height = int(tokens[1])
        max_val = int(tokens[2])

        if max_val <= 255:
            bits = 8
            nbytes = width * height
        else:
            bits = 16
            nbytes = width * height * 2

        raw_data = f.read(nbytes)
        if len(raw_data) < nbytes:
            raise ValueError(
                f"PGM-Datei zu kurz: erwartet {nbytes} Bytes, bekam {len(raw_data)}")

    # 16-bit PGM ist Big-Endian → zu Little-Endian konvertieren
    if bits == 16:
        pixels = np.frombuffer(raw_data, dtype='>u2')
        pixel_data = pixels.astype('<u2').tobytes()
    else:
        pixel_data = raw_data

    return PGMData(
        width=width,
        height=height,
        max_val=max_val,
        bits_per_sample=bits,
        pixel_data=pixel_data,
    )


# ── DNG Writer ───────────────────────────────────────────────

class DNGWriter:
    """Erzeugt DNG-Dateien aus Rohdaten."""

    def write(self, filepath: str, pgm: PGMData, config: DNGConfig):
        """Schreibt eine DNG-Datei."""
        white_level = config.white_level if config.white_level is not None else pgm.max_val
        crop_w, crop_h = config.default_crop_size or (pgm.width, pgm.height)
        is_mono = config.cfa_pattern == "MONO"

        tags = []

        # ── Standard TIFF Image Tags ──
        tags.append((TAG_IMAGE_WIDTH, TIFF_LONG, 1,
                     struct.pack('<I', pgm.width)))
        tags.append((TAG_IMAGE_LENGTH, TIFF_LONG, 1,
                     struct.pack('<I', pgm.height)))
        tags.append((TAG_BITS_PER_SAMPLE, TIFF_SHORT, 1,
                     struct.pack('<H', pgm.bits_per_sample)))
        tags.append((TAG_COMPRESSION, TIFF_SHORT, 1,
                     struct.pack('<H', 1)))  # Uncompressed
        tags.append((TAG_PHOTOMETRIC_INTERPRETATION, TIFF_SHORT, 1,
                     struct.pack('<H', PI_LINEAR_RAW if is_mono else PI_CFA)))
        tags.append((TAG_SAMPLES_PER_PIXEL, TIFF_SHORT, 1,
                     struct.pack('<H', 1)))
        tags.append((TAG_ROWS_PER_STRIP, TIFF_LONG, 1,
                     struct.pack('<I', pgm.height)))

        strip_byte_count = len(pgm.pixel_data)
        tags.append((TAG_STRIP_BYTE_COUNTS, TIFF_LONG, 1,
                     struct.pack('<I', strip_byte_count)))

        # ── DNG Version ──
        tags.append((TAG_DNG_VERSION, TIFF_BYTE, 4,
                     bytes(config.dng_version)))
        tags.append((TAG_DNG_BACKWARD_VERSION, TIFF_BYTE, 4,
                     bytes(config.dng_backward_version)))

        # ── UniqueCameraModel ──
        model_bytes = config.camera_model.encode('ascii', errors='replace') + b'\x00'
        tags.append((TAG_UNIQUE_CAMERA_MODEL, TIFF_ASCII, len(model_bytes), model_bytes))

        # ── CFA Tags (nur bei Bayer) ──
        if not is_mono:
            tags.append((TAG_CFA_REPEAT_PATTERN_DIM, TIFF_SHORT, 2,
                         struct.pack('<HH', 2, 2)))

            pattern_bytes = CFA_PATTERNS.get(config.cfa_pattern)
            if pattern_bytes is None:
                raise ValueError(
                    f"Unbekanntes CFA-Pattern: {config.cfa_pattern}. "
                    f"Erlaubt: {', '.join(CFA_PATTERNS.keys())}, MONO")
            tags.append((TAG_CFA_PATTERN_TIFFEP, TIFF_BYTE, 4, pattern_bytes))

            tags.append((TAG_CFA_PLANE_COLOR, TIFF_BYTE, 3,
                         bytes([0, 1, 2])))  # R, G, B
            tags.append((TAG_CFA_LAYOUT, TIFF_SHORT, 1,
                         struct.pack('<H', 1)))  # Rectangular

        # ── Black/White Level ──
        tags.append((TAG_BLACK_LEVEL, TIFF_LONG, 1,
                     struct.pack('<I', config.black_level)))
        tags.append((TAG_WHITE_LEVEL, TIFF_LONG, 1,
                     struct.pack('<I', white_level)))

        # ── Default Scale ──
        scale_data = struct.pack('<IIII',
                                 config.default_scale[0], 1,
                                 config.default_scale[1], 1)
        tags.append((TAG_DEFAULT_SCALE, TIFF_RATIONAL, 2, scale_data))

        # ── Default Crop ──
        crop_origin_data = struct.pack('<IIII',
                                       config.default_crop_origin[0], 1,
                                       config.default_crop_origin[1], 1)
        tags.append((TAG_DEFAULT_CROP_ORIGIN, TIFF_RATIONAL, 2, crop_origin_data))

        crop_size_data = struct.pack('<IIII', crop_w, 1, crop_h, 1)
        tags.append((TAG_DEFAULT_CROP_SIZE, TIFF_RATIONAL, 2, crop_size_data))

        # ── Farbmatrizen ──
        if config.color_matrix_1 is not None:
            tags.append((TAG_CALIBRATION_ILLUMINANT_1, TIFF_SHORT, 1,
                         struct.pack('<H', config.illuminant_1)))
            mat_data = self._encode_matrix(config.color_matrix_1)
            tags.append((TAG_COLOR_MATRIX_1, TIFF_SRATIONAL, 9, mat_data))

        if config.color_matrix_2 is not None:
            tags.append((TAG_CALIBRATION_ILLUMINANT_2, TIFF_SHORT, 1,
                         struct.pack('<H', config.illuminant_2)))
            mat_data = self._encode_matrix(config.color_matrix_2)
            tags.append((TAG_COLOR_MATRIX_2, TIFF_SRATIONAL, 9, mat_data))

        # ── Forward-Matrizen ──
        if config.forward_matrix_1 is not None:
            tags.append((TAG_FORWARD_MATRIX_1, TIFF_SRATIONAL, 9,
                         self._encode_matrix(config.forward_matrix_1)))
        if config.forward_matrix_2 is not None:
            tags.append((TAG_FORWARD_MATRIX_2, TIFF_SRATIONAL, 9,
                         self._encode_matrix(config.forward_matrix_2)))

        # ── AsShotNeutral (Weißabgleich) ──
        if config.as_shot_neutral is not None:
            r, g, b = config.as_shot_neutral
            neutral_data = struct.pack('<IIIIII',
                                       int(round(r * 10000)), 10000,
                                       int(round(g * 10000)), 10000,
                                       int(round(b * 10000)), 10000)
            tags.append((TAG_AS_SHOT_NEUTRAL, TIFF_RATIONAL, 3, neutral_data))

        # ── BaselineExposure ──
        if config.baseline_exposure != 0.0:
            num, den = _float_to_srational(config.baseline_exposure)
            tags.append((TAG_BASELINE_EXPOSURE, TIFF_SRATIONAL, 1,
                         struct.pack('<ii', num, den)))

        # ── ProfileName ──
        if config.profile_name:
            name_bytes = config.profile_name.encode('ascii', errors='replace') + b'\x00'
            tags.append((TAG_PROFILE_NAME, TIFF_ASCII, len(name_bytes), name_bytes))

        # ── Tags sortieren (TIFF-Pflicht) ──
        tags.sort(key=lambda t: t[0])

        # ── Berechne IFD-Layout ohne StripOffsets ──
        # StripOffsets wird als LONG (4 Bytes, inline) eingefügt
        num_entries = len(tags) + 1  # +1 für StripOffsets
        header_size = 8
        ifd_size = 2 + (num_entries * 12) + 4

        # Extra-Daten berechnen
        extra_data_size = 0
        for _, tag_type, count, data in tags:
            type_size = TIFF_TYPE_SIZE.get(tag_type, 1)
            total_size = count * type_size
            if total_size > 4:
                extra_data_size += len(data)
                if extra_data_size % 2 != 0:
                    extra_data_size += 1

        strip_offset = header_size + ifd_size + extra_data_size

        # StripOffsets in die Tag-Liste einfügen
        tags.append((TAG_STRIP_OFFSETS, TIFF_LONG, 1,
                     struct.pack('<I', strip_offset)))
        tags.sort(key=lambda t: t[0])

        # ── Datei zusammenbauen ──
        data_offset = header_size + ifd_size

        out = bytearray()
        # TIFF Header
        out += struct.pack('<HHI', 0x4949, TIFF_MAGIC, 8)
        # IFD Entry Count
        out += struct.pack('<H', num_entries)

        extra_data = bytearray()
        ifd_entries = bytearray()

        for tag_id, tag_type, count, data in tags:
            type_size = TIFF_TYPE_SIZE.get(tag_type, 1)
            total_size = count * type_size

            if total_size <= 4:
                value_bytes = data[:total_size]
                value_bytes = value_bytes + b'\x00' * (4 - len(value_bytes))
                ifd_entries += struct.pack('<HHI', tag_id, tag_type, count)
                ifd_entries += value_bytes
            else:
                offset = data_offset + len(extra_data)
                ifd_entries += struct.pack('<HHI', tag_id, tag_type, count)
                ifd_entries += struct.pack('<I', offset)
                extra_data += data
                if len(extra_data) % 2 != 0:
                    extra_data += b'\x00'

        out += ifd_entries
        out += struct.pack('<I', 0)  # Next IFD offset (none)
        out += extra_data
        out += pgm.pixel_data

        with open(filepath, 'wb') as f:
            f.write(out)

    def _encode_matrix(self, matrix: np.ndarray) -> bytes:
        """Kodiert eine 3x3-Matrix als 9 SRATIONAL-Werte."""
        flat = matrix.flatten()
        data = bytearray()
        for val in flat:
            num, den = _float_to_srational(float(val))
            data += struct.pack('<ii', num, den)
        return bytes(data)


# ── Hilfsfunktionen ──────────────────────────────────────────

def config_from_camera_info(cam) -> DNGConfig:
    """Erstellt eine DNGConfig aus einem CameraInfo-Objekt."""
    return DNGConfig(
        camera_model=cam.unique_camera_model,
        color_matrix_1=cam.color_matrix_a,
        color_matrix_2=cam.color_matrix_d65,
        illuminant_1=ILLUMINANT_A,
        illuminant_2=ILLUMINANT_D65,
    )


def pgm_to_dng(pgm_path: str, dng_path: str,
               config: Optional[DNGConfig] = None) -> str:
    """
    Konvertiert eine PGM-Datei zu DNG.

    Args:
        pgm_path: Pfad zur PGM-Eingabedatei
        dng_path: Pfad für die DNG-Ausgabedatei
        config: DNG-Konfiguration (optional, nutzt Defaults)

    Returns:
        Pfad zur erzeugten DNG-Datei
    """
    if config is None:
        config = DNGConfig()
    pgm = read_pgm(pgm_path)
    DNGWriter().write(dng_path, pgm, config)
    return dng_path


def create_dng_from_array(pixel_data: np.ndarray, dng_path: str,
                          config: Optional[DNGConfig] = None) -> str:
    """
    Erzeugt eine DNG-Datei aus einem numpy-Array.

    Args:
        pixel_data: 2D numpy-Array (height × width), uint8 oder uint16
        dng_path: Pfad für die DNG-Ausgabedatei
        config: DNG-Konfiguration (optional)

    Returns:
        Pfad zur erzeugten DNG-Datei
    """
    if config is None:
        config = DNGConfig()
    if pixel_data.ndim != 2:
        raise ValueError(f"Erwartet 2D-Array, bekam {pixel_data.ndim}D")

    height, width = pixel_data.shape

    if pixel_data.dtype == np.uint8:
        bits = 8
        max_val = 255
        data_bytes = pixel_data.tobytes()
    elif pixel_data.dtype == np.uint16:
        bits = 16
        max_val = 65535
        le_data = pixel_data.astype('<u2')
        data_bytes = le_data.tobytes()
    else:
        raise ValueError(f"Erwartet uint8 oder uint16, bekam {pixel_data.dtype}")

    pgm = PGMData(
        width=width,
        height=height,
        max_val=max_val,
        bits_per_sample=bits,
        pixel_data=data_bytes,
    )
    DNGWriter().write(dng_path, pgm, config)
    return dng_path


# ── CLI ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Verwendung: python dng_writer.py <eingabe.pgm> <ausgabe.dng> [optionen]")
        print()
        print("Optionen:")
        print("  --pattern=RGGB    CFA-Pattern (RGGB, BGGR, GRBG, GBRG, MONO)")
        print("  --camera=NAME     Kameramodell-Name")
        print("  --black=N         Black-Level (Standard: 0)")
        print("  --white=N         White-Level (Standard: max aus PGM)")
        print("  --wp=R,G,B        Weißabgleich Neutral-Werte")
        sys.exit(1)

    pgm_path = sys.argv[1]
    dng_path = sys.argv[2]

    config = DNGConfig()

    for arg in sys.argv[3:]:
        if arg.startswith('--pattern='):
            config.cfa_pattern = arg.split('=', 1)[1].upper()
        elif arg.startswith('--camera='):
            config.camera_model = arg.split('=', 1)[1]
        elif arg.startswith('--black='):
            config.black_level = int(arg.split('=', 1)[1])
        elif arg.startswith('--white='):
            config.white_level = int(arg.split('=', 1)[1])
        elif arg.startswith('--wp='):
            vals = arg.split('=', 1)[1].split(',')
            if len(vals) == 3:
                floats = [float(v) for v in vals]
                mx = max(floats)
                config.as_shot_neutral = tuple(v / mx for v in floats)

    print(f"Lese: {pgm_path}")
    pgm = read_pgm(pgm_path)
    print(f"  {pgm.width}×{pgm.height}, {pgm.bits_per_sample}-bit, maxval={pgm.max_val}")

    DNGWriter().write(dng_path, pgm, config)
    print(f"DNG geschrieben: {dng_path}")
    print(f"  Kamera: {config.camera_model}")
    print(f"  CFA: {config.cfa_pattern}")
    print(f"  Black: {config.black_level}, White: {config.white_level or pgm.max_val}")
