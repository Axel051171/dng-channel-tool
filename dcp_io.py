"""
DCP (DNG Camera Profile) Reader/Writer

DCP files are TIFF-based files containing color matrices and other
profile data used by Adobe Camera Raw and Lightroom.

Tag IDs verified against real Adobe DCP files.
"""

import struct
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np


# TIFF Types
TIFF_BYTE = 1
TIFF_ASCII = 2
TIFF_SHORT = 3
TIFF_LONG = 4
TIFF_RATIONAL = 5
TIFF_SRATIONAL = 10
TIFF_FLOAT = 11     # IEEE 754 single-precision
TIFF_DOUBLE = 12     # IEEE 754 double-precision

# DCP/DNG Tag IDs (verified from real Adobe DCP files)
TAG_UNIQUE_CAMERA_MODEL = 50708          # 0xC614 - ASCII
TAG_COLOR_MATRIX_1 = 50721              # 0xC621 - SRATIONAL[9]
TAG_COLOR_MATRIX_2 = 50722              # 0xC622 - SRATIONAL[9]
TAG_CALIBRATION_ILLUMINANT_1 = 50778    # 0xC65A - SHORT
TAG_CALIBRATION_ILLUMINANT_2 = 50779    # 0xC65B - SHORT
TAG_PROFILE_CALIBRATION_SIGNATURE = 50932  # 0xC6F4 - ASCII
TAG_PROFILE_NAME = 50936                # 0xC6F8 - ASCII
TAG_PROFILE_HUE_SAT_MAP_DIMS = 50937   # 0xC6F9 - LONG[3]
TAG_PROFILE_HUE_SAT_MAP_DATA_1 = 50938 # 0xC6FA - FLOAT[]
TAG_PROFILE_HUE_SAT_MAP_DATA_2 = 50939 # 0xC6FB - FLOAT[]
TAG_PROFILE_TONE_CURVE = 50940          # 0xC6FC - FLOAT[]
TAG_PROFILE_EMBED_POLICY = 50941        # 0xC6FD - LONG
TAG_PROFILE_COPYRIGHT = 50942           # 0xC6FE - ASCII
TAG_FORWARD_MATRIX_1 = 50964           # 0xC714 - SRATIONAL[9]
TAG_FORWARD_MATRIX_2 = 50965           # 0xC715 - SRATIONAL[9]
TAG_PROFILE_LOOK_TABLE_DIMS = 50981    # 0xC725 - LONG[3]
TAG_PROFILE_LOOK_TABLE_DATA = 50982    # 0xC726 - FLOAT[]
TAG_REDUCTION_MATRIX_1 = 50725         # 0xC625 - SRATIONAL[]
TAG_REDUCTION_MATRIX_2 = 50726         # 0xC626 - SRATIONAL[]
TAG_PROFILE_HUE_SAT_MAP_ENCODING = 51107  # 0xC7A3 - LONG
TAG_PROFILE_LOOK_TABLE_ENCODING = 51108   # 0xC7A4 - LONG
TAG_BASELINE_EXPOSURE_OFFSET = 51109      # 0xC7A5 - SRATIONAL
TAG_DEFAULT_BLACK_RENDER = 51110          # 0xC7A6 - LONG
TAG_UNIQUE_CAMERA_MODEL_RESTRICTION = 50708  # Same as UniqueCameraModel in profile context

# Illuminant constants
ILLUMINANT_A = 17           # Standard Light A (~2856K)
ILLUMINANT_D50 = 23         # D50 (~5003K)
ILLUMINANT_D55 = 20         # D55 (~5503K)
ILLUMINANT_D65 = 21         # D65 (~6504K)
ILLUMINANT_D75 = 22         # D75 (~7504K)

ILLUMINANT_NAMES = {
    17: "Standard Light A",
    20: "D55",
    21: "D65",
    22: "D75",
    23: "D50",
}

# DCP uses magic 0x4352 ("CR" = Camera Raw) instead of TIFF's 42
DCP_MAGIC = 0x4352

TIFF_TYPE_SIZE = {
    TIFF_BYTE: 1,
    TIFF_ASCII: 1,
    TIFF_SHORT: 2,
    TIFF_LONG: 4,
    TIFF_RATIONAL: 8,
    TIFF_SRATIONAL: 8,
    TIFF_FLOAT: 4,
    TIFF_DOUBLE: 8,
}


@dataclass
class DCPProfile:
    """Represents a DNG Camera Profile."""
    camera_model: str = ""
    profile_name: str = "Custom Profile"
    copyright: str = "DNG Channel Tool"
    embed_policy: int = 0  # 0 = allow embedding
    calibration_signature: str = ""

    # Color matrices (3x3, XYZ to Camera RGB)
    color_matrix_1: Optional[np.ndarray] = None
    color_matrix_2: Optional[np.ndarray] = None

    # Forward matrices (3x3, Camera RGB to XYZ D50)
    forward_matrix_1: Optional[np.ndarray] = None
    forward_matrix_2: Optional[np.ndarray] = None

    # Illuminants
    illuminant_1: int = ILLUMINANT_A
    illuminant_2: int = ILLUMINANT_D65

    # HueSatMap (preserved for pass-through)
    hue_sat_map_dims: Optional[tuple] = None
    hue_sat_map_data_1: Optional[bytes] = None
    hue_sat_map_data_2: Optional[bytes] = None

    # LookTable (preserved for pass-through)
    look_table_dims: Optional[tuple] = None
    look_table_data: Optional[bytes] = None

    # Reduction matrices (optional, for >3 color channels)
    reduction_matrix_1: Optional[np.ndarray] = None
    reduction_matrix_2: Optional[np.ndarray] = None

    # Tone curve (raw float data, preserved for pass-through)
    tone_curve_data: Optional[bytes] = None
    tone_curve_count: int = 0

    # Extended DCP fields (from dcpTool)
    hue_sat_map_encoding: int = 0       # 0 = Linear, 1 = sRGB
    look_table_encoding: int = 0        # 0 = Linear, 1 = sRGB
    baseline_exposure_offset: float = 0.0
    default_black_render: int = 0       # 0 = Auto, 1 = None

    def has_dual_illuminant(self) -> bool:
        return self.color_matrix_2 is not None


def _float_to_srational(value: float, denominator: int = 10000) -> tuple:
    """Convert float to SRATIONAL (numerator, denominator)."""
    numerator = int(round(value * denominator))
    return (numerator, denominator)


def _srational_to_float(num: int, den: int) -> float:
    """Convert SRATIONAL to float."""
    if den == 0:
        return 0.0
    return num / den


class DCPWriter:
    """Writes DCP (DNG Camera Profile) files."""

    def write(self, filepath: str, profile: DCPProfile):
        """Write a DCP profile to file."""
        tags = []

        # UniqueCameraModel (required)
        model_bytes = profile.camera_model.encode('ascii', errors='replace') + b'\x00'
        tags.append((TAG_UNIQUE_CAMERA_MODEL, TIFF_ASCII, len(model_bytes), model_bytes))

        # CalibrationIlluminant1 (required)
        tags.append((TAG_CALIBRATION_ILLUMINANT_1, TIFF_SHORT, 1,
                      struct.pack('<H', profile.illuminant_1)))

        # ColorMatrix1 (required)
        if profile.color_matrix_1 is not None:
            mat_data = self._encode_matrix(profile.color_matrix_1)
            tags.append((TAG_COLOR_MATRIX_1, TIFF_SRATIONAL, 9, mat_data))

        # ColorMatrix2 + CalibrationIlluminant2 (optional)
        if profile.color_matrix_2 is not None:
            tags.append((TAG_CALIBRATION_ILLUMINANT_2, TIFF_SHORT, 1,
                          struct.pack('<H', profile.illuminant_2)))
            mat_data = self._encode_matrix(profile.color_matrix_2)
            tags.append((TAG_COLOR_MATRIX_2, TIFF_SRATIONAL, 9, mat_data))

        # ProfileCalibrationSignature (optional)
        if profile.calibration_signature:
            sig_bytes = profile.calibration_signature.encode('ascii', errors='replace') + b'\x00'
            tags.append((TAG_PROFILE_CALIBRATION_SIGNATURE, TIFF_ASCII, len(sig_bytes), sig_bytes))

        # ForwardMatrix1 (optional)
        if profile.forward_matrix_1 is not None:
            mat_data = self._encode_matrix(profile.forward_matrix_1)
            tags.append((TAG_FORWARD_MATRIX_1, TIFF_SRATIONAL, 9, mat_data))

        # ForwardMatrix2 (optional)
        if profile.forward_matrix_2 is not None:
            mat_data = self._encode_matrix(profile.forward_matrix_2)
            tags.append((TAG_FORWARD_MATRIX_2, TIFF_SRATIONAL, 9, mat_data))

        # ProfileName
        name_bytes = profile.profile_name.encode('ascii', errors='replace') + b'\x00'
        tags.append((TAG_PROFILE_NAME, TIFF_ASCII, len(name_bytes), name_bytes))

        # ProfileHueSatMapDims + Data (pass-through)
        if profile.hue_sat_map_dims is not None:
            dims_data = struct.pack('<III', *profile.hue_sat_map_dims)
            tags.append((TAG_PROFILE_HUE_SAT_MAP_DIMS, TIFF_LONG, 3, dims_data))
        if profile.hue_sat_map_data_1 is not None:
            count = len(profile.hue_sat_map_data_1) // 4  # FLOAT = 4 bytes
            tags.append((TAG_PROFILE_HUE_SAT_MAP_DATA_1, TIFF_FLOAT, count, profile.hue_sat_map_data_1))
        if profile.hue_sat_map_data_2 is not None:
            count = len(profile.hue_sat_map_data_2) // 4
            tags.append((TAG_PROFILE_HUE_SAT_MAP_DATA_2, TIFF_FLOAT, count, profile.hue_sat_map_data_2))

        # ProfileToneCurve (pass-through)
        if profile.tone_curve_data is not None:
            count = len(profile.tone_curve_data) // 4  # FLOAT = 4 bytes
            tags.append((TAG_PROFILE_TONE_CURVE, TIFF_FLOAT, count, profile.tone_curve_data))

        # ProfileEmbedPolicy
        tags.append((TAG_PROFILE_EMBED_POLICY, TIFF_LONG, 1,
                      struct.pack('<I', profile.embed_policy)))

        # ProfileCopyright
        cr_bytes = profile.copyright.encode('ascii', errors='replace') + b'\x00'
        tags.append((TAG_PROFILE_COPYRIGHT, TIFF_ASCII, len(cr_bytes), cr_bytes))

        # ProfileLookTableDims + Data (pass-through)
        if profile.look_table_dims is not None:
            dims_data = struct.pack('<III', *profile.look_table_dims)
            tags.append((TAG_PROFILE_LOOK_TABLE_DIMS, TIFF_LONG, 3, dims_data))
        if profile.look_table_data is not None:
            count = len(profile.look_table_data) // 4
            tags.append((TAG_PROFILE_LOOK_TABLE_DATA, TIFF_FLOAT, count, profile.look_table_data))

        # ReductionMatrix1/2 (optional)
        if profile.reduction_matrix_1 is not None:
            rows, cols = profile.reduction_matrix_1.shape
            mat_data = self._encode_matrix_generic(profile.reduction_matrix_1)
            tags.append((TAG_REDUCTION_MATRIX_1, TIFF_SRATIONAL, rows * cols, mat_data))
        if profile.reduction_matrix_2 is not None:
            rows, cols = profile.reduction_matrix_2.shape
            mat_data = self._encode_matrix_generic(profile.reduction_matrix_2)
            tags.append((TAG_REDUCTION_MATRIX_2, TIFF_SRATIONAL, rows * cols, mat_data))

        # HueSatMapEncoding (only if non-linear)
        if profile.hue_sat_map_encoding != 0:
            tags.append((TAG_PROFILE_HUE_SAT_MAP_ENCODING, TIFF_LONG, 1,
                         struct.pack('<I', profile.hue_sat_map_encoding)))

        # LookTableEncoding (only if non-linear)
        if profile.look_table_encoding != 0:
            tags.append((TAG_PROFILE_LOOK_TABLE_ENCODING, TIFF_LONG, 1,
                         struct.pack('<I', profile.look_table_encoding)))

        # BaselineExposureOffset (only if non-zero)
        if profile.baseline_exposure_offset != 0.0:
            num, den = _float_to_srational(profile.baseline_exposure_offset)
            tags.append((TAG_BASELINE_EXPOSURE_OFFSET, TIFF_SRATIONAL, 1,
                         struct.pack('<ii', num, den)))

        # DefaultBlackRender (only if non-auto)
        if profile.default_black_render != 0:
            tags.append((TAG_DEFAULT_BLACK_RENDER, TIFF_LONG, 1,
                         struct.pack('<I', profile.default_black_render)))

        # Sort tags by tag ID (TIFF requirement)
        tags.sort(key=lambda t: t[0])

        # Calculate layout
        header_size = 8
        num_entries = len(tags)
        ifd_size = 2 + (num_entries * 12) + 4
        data_offset = header_size + ifd_size

        # Build output
        out = bytearray()
        out += struct.pack('<HHI', 0x4949, DCP_MAGIC, 8)
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
                # Pad to word boundary
                if len(extra_data) % 2 != 0:
                    extra_data += b'\x00'

        out += ifd_entries
        out += struct.pack('<I', 0)  # Next IFD offset (none)
        out += extra_data

        with open(filepath, 'wb') as f:
            f.write(out)

    def _encode_matrix(self, matrix: np.ndarray) -> bytes:
        """Encode a 3x3 matrix as 9 SRATIONAL values."""
        return self._encode_matrix_generic(matrix)

    def _encode_matrix_generic(self, matrix: np.ndarray) -> bytes:
        """Encode a matrix of any shape as SRATIONAL values."""
        flat = matrix.flatten()
        data = bytearray()
        for val in flat:
            num, den = _float_to_srational(float(val))
            data += struct.pack('<ii', num, den)
        return bytes(data)


class DCPReader:
    """Reads DCP (DNG Camera Profile) files."""

    def read(self, filepath: str) -> DCPProfile:
        """Read a DCP profile from file."""
        with open(filepath, 'rb') as f:
            data = f.read()

        if len(data) < 8:
            raise ValueError(f"Datei zu kurz für TIFF-Header ({len(data)} Bytes)")

        profile = DCPProfile()

        # Parse TIFF header
        byte_order = struct.unpack_from('<H', data, 0)[0]
        if byte_order == 0x4949:
            endian = '<'
        elif byte_order == 0x4D4D:
            endian = '>'
        else:
            raise ValueError(f"Invalid byte order: {byte_order:#x}")

        magic = struct.unpack_from(f'{endian}H', data, 2)[0]
        if magic != 42 and magic != DCP_MAGIC:
            raise ValueError(f"Invalid TIFF/DCP magic: {magic}")

        ifd_offset = struct.unpack_from(f'{endian}I', data, 4)[0]

        if ifd_offset + 2 > len(data):
            raise ValueError(f"IFD-Offset {ifd_offset} liegt außerhalb der Datei ({len(data)} Bytes)")

        # Parse IFD
        num_entries = struct.unpack_from(f'{endian}H', data, ifd_offset)[0]

        if num_entries > 1000:
            raise ValueError(f"Unrealistische Anzahl IFD-Einträge: {num_entries}")

        offset = ifd_offset + 2

        for i in range(num_entries):
            if offset + 12 > len(data):
                break  # Datei abgeschnitten
            tag_id, tag_type, count = struct.unpack_from(f'{endian}HHI', data, offset)
            value_offset_raw = data[offset + 8:offset + 12]
            offset += 12

            type_size = TIFF_TYPE_SIZE.get(tag_type, 1)
            total_size = count * type_size

            if total_size <= 4:
                value_data = value_offset_raw
            else:
                val_offset = struct.unpack_from(f'{endian}I', value_offset_raw, 0)[0]
                if val_offset + total_size > len(data):
                    continue  # Offset außerhalb der Datei, Tag überspringen
                value_data = data[val_offset:val_offset + total_size]

            # Parse known tags
            if tag_id == TAG_UNIQUE_CAMERA_MODEL:
                profile.camera_model = value_data[:count].decode('ascii', errors='replace').rstrip('\x00')

            elif tag_id == TAG_PROFILE_NAME:
                profile.profile_name = value_data[:count].decode('ascii', errors='replace').rstrip('\x00')

            elif tag_id == TAG_PROFILE_COPYRIGHT:
                profile.copyright = value_data[:count].decode('ascii', errors='replace').rstrip('\x00')

            elif tag_id == TAG_PROFILE_CALIBRATION_SIGNATURE:
                profile.calibration_signature = value_data[:count].decode('ascii', errors='replace').rstrip('\x00')

            elif tag_id == TAG_PROFILE_EMBED_POLICY:
                profile.embed_policy = struct.unpack_from(f'{endian}I', value_data, 0)[0]

            elif tag_id == TAG_CALIBRATION_ILLUMINANT_1:
                profile.illuminant_1 = struct.unpack_from(f'{endian}H', value_data, 0)[0]

            elif tag_id == TAG_CALIBRATION_ILLUMINANT_2:
                profile.illuminant_2 = struct.unpack_from(f'{endian}H', value_data, 0)[0]

            elif tag_id == TAG_COLOR_MATRIX_1:
                profile.color_matrix_1 = self._decode_matrix(value_data, endian)

            elif tag_id == TAG_COLOR_MATRIX_2:
                profile.color_matrix_2 = self._decode_matrix(value_data, endian)

            elif tag_id == TAG_FORWARD_MATRIX_1:
                profile.forward_matrix_1 = self._decode_matrix(value_data, endian)

            elif tag_id == TAG_FORWARD_MATRIX_2:
                profile.forward_matrix_2 = self._decode_matrix(value_data, endian)

            # Pass-through data (preserved as raw bytes)
            elif tag_id == TAG_PROFILE_HUE_SAT_MAP_DIMS:
                d = struct.unpack_from(f'{endian}III', value_data, 0)
                profile.hue_sat_map_dims = d

            elif tag_id == TAG_PROFILE_HUE_SAT_MAP_DATA_1:
                profile.hue_sat_map_data_1 = value_data[:total_size]

            elif tag_id == TAG_PROFILE_HUE_SAT_MAP_DATA_2:
                profile.hue_sat_map_data_2 = value_data[:total_size]

            elif tag_id == TAG_PROFILE_TONE_CURVE:
                profile.tone_curve_data = value_data[:total_size]
                profile.tone_curve_count = count

            elif tag_id == TAG_PROFILE_LOOK_TABLE_DIMS:
                d = struct.unpack_from(f'{endian}III', value_data, 0)
                profile.look_table_dims = d

            elif tag_id == TAG_PROFILE_LOOK_TABLE_DATA:
                profile.look_table_data = value_data[:total_size]

            elif tag_id == TAG_REDUCTION_MATRIX_1:
                profile.reduction_matrix_1 = self._decode_matrix_generic(
                    value_data, endian, count)

            elif tag_id == TAG_REDUCTION_MATRIX_2:
                profile.reduction_matrix_2 = self._decode_matrix_generic(
                    value_data, endian, count)

            elif tag_id == TAG_PROFILE_HUE_SAT_MAP_ENCODING:
                profile.hue_sat_map_encoding = struct.unpack_from(
                    f'{endian}I', value_data, 0)[0]

            elif tag_id == TAG_PROFILE_LOOK_TABLE_ENCODING:
                profile.look_table_encoding = struct.unpack_from(
                    f'{endian}I', value_data, 0)[0]

            elif tag_id == TAG_BASELINE_EXPOSURE_OFFSET:
                num, den = struct.unpack_from(f'{endian}ii', value_data, 0)
                profile.baseline_exposure_offset = _srational_to_float(num, den)

            elif tag_id == TAG_DEFAULT_BLACK_RENDER:
                profile.default_black_render = struct.unpack_from(
                    f'{endian}I', value_data, 0)[0]

        return profile

    def _decode_matrix(self, data: bytes, endian: str) -> np.ndarray:
        """Decode 9 SRATIONAL values into a 3x3 matrix."""
        return self._decode_matrix_generic(data, endian, 9).reshape(3, 3)

    def _decode_matrix_generic(self, data: bytes, endian: str, count: int) -> np.ndarray:
        """Decode SRATIONAL values into a matrix. Infers shape from count."""
        values = []
        for i in range(count):
            num, den = struct.unpack_from(f'{endian}ii', data, i * 8)
            values.append(_srational_to_float(num, den))
        arr = np.array(values)
        # Try to reshape: 9→3x3, 12→3x4 or 4x3, etc.
        if count == 9:
            return arr.reshape(3, 3)
        elif count % 3 == 0:
            return arr.reshape(-1, 3)
        return arr


def get_adobe_profile_dir() -> str:
    """Get the Adobe CameraRaw camera profiles directory."""
    appdata = os.environ.get('APPDATA', '')
    if appdata:
        return os.path.join(appdata, 'Adobe', 'CameraRaw', 'CameraProfiles')
    home = os.path.expanduser('~')
    return os.path.join(home, 'AppData', 'Roaming', 'Adobe', 'CameraRaw', 'CameraProfiles')


def rewrite_dcp_camera_model(src_path: str, dest_path: str,
                              new_camera_model: str,
                              new_profile_name: str = None) -> DCPProfile:
    """
    Liest ein DCP-Profil, ändert das Kameramodell (UniqueCameraModel)
    und optional den Profilnamen, und schreibt es neu.

    Damit kann ein bestehendes DCP-Profil für eine andere Kamera
    kompatibel gemacht werden (z.B. Z6 → Z6_2 oder umgekehrt).

    Args:
        src_path: Quell-DCP-Datei
        dest_path: Ziel-DCP-Datei (kann gleich src_path sein)
        new_camera_model: Neuer Kamera-Modellname (exakt wie in EXIF)
        new_profile_name: Optional neuer Profilname

    Returns:
        Das modifizierte DCPProfile
    """
    reader = DCPReader()
    profile = reader.read(src_path)

    old_model = profile.camera_model
    profile.camera_model = new_camera_model

    if new_profile_name:
        profile.profile_name = new_profile_name
    elif old_model in profile.profile_name:
        # Profilname automatisch anpassen wenn alter Kameraname drin steckt
        profile.profile_name = profile.profile_name.replace(old_model, new_camera_model)

    writer = DCPWriter()
    writer.write(dest_path, profile)

    return profile


def install_dcp_to_adobe(dcp_path: str, subfolder: str = "Channel Swap") -> str:
    """Copy a DCP file to the Adobe CameraRaw profile directory."""
    import shutil
    dest_dir = os.path.join(get_adobe_profile_dir(), subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(dcp_path))
    shutil.copy2(dcp_path, dest_path)
    return dest_path
