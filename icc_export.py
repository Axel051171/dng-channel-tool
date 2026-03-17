"""
ICC v2 color profile (.icc / .icm) generation from channel swaps,
tone curves, and color transformations.

Generates Display-class ('mntr') matrix/TRC profiles that encode:
  - A 3x3 color matrix via rXYZ / gXYZ / bXYZ colorant tags
  - Per-channel Tone Response Curves (rTRC / gTRC / bTRC)
  - D50 media white point

All multi-byte values are written big-endian per the ICC specification.
XYZ values use s15Fixed16Number encoding (value * 65536).
"""

import os
import struct
import datetime
import hashlib
import numpy as np
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# D50 white point in XYZ (ICC PCS illuminant)
D50_WHITE = (0.9642, 1.0, 0.8249)

# sRGB colorant matrix columns (XYZ values for R, G, B primaries under D50)
SRGB_MATRIX_D50 = np.array([
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
])

ICC_SIGNATURE = b'acsp'
ICC_VERSION_2_4 = 0x02400000


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _s15fixed16(value: float) -> bytes:
    """Encode a float as an ICC s15Fixed16Number (4 bytes, big-endian)."""
    fixed = int(round(value * 65536.0))
    # Clamp to signed 32-bit range
    fixed = max(-2147483648, min(2147483647, fixed))
    return struct.pack('>i', fixed)


def _u16(value: int) -> bytes:
    """Encode an unsigned 16-bit integer (big-endian)."""
    return struct.pack('>H', max(0, min(65535, value)))


def _u32(value: int) -> bytes:
    """Encode an unsigned 32-bit integer (big-endian)."""
    return struct.pack('>I', value)


def _pad4(data: bytes) -> bytes:
    """Pad *data* to a 4-byte boundary with null bytes."""
    remainder = len(data) % 4
    if remainder:
        data += b'\x00' * (4 - remainder)
    return data


# ---------------------------------------------------------------------------
# Tag data builders
# ---------------------------------------------------------------------------

def _build_xyz_type(x: float, y: float, z: float) -> bytes:
    """Build an XYZType tag element.

    Layout: 'XYZ ' (4) + reserved (4) + X(4) + Y(4) + Z(4) = 20 bytes.
    """
    return b'XYZ ' + b'\x00' * 4 + _s15fixed16(x) + _s15fixed16(y) + _s15fixed16(z)


def _build_curve_type(values: Union[List[float], float, None]) -> bytes:
    """Build a curveType tag element.

    Parameters
    ----------
    values : list of float | float | None
        - ``None`` or empty list: identity curve (count = 0).
        - Single float: interpreted as a gamma exponent encoded as
          u8Fixed8Number (count = 1).
        - List of 2+ floats (0.0 - 1.0): LUT entries encoded as uint16
          (count = len(values)).
    """
    sig = b'curv'
    reserved = b'\x00' * 4

    # Identity
    if values is None or (isinstance(values, list) and len(values) == 0):
        return sig + reserved + _u32(0)

    # Single gamma value
    if isinstance(values, (int, float)):
        gamma = float(values)
        # u8Fixed8Number: integer part in high byte, fractional * 256 in low byte
        gamma_fixed = int(round(gamma * 256.0))
        gamma_fixed = max(0, min(65535, gamma_fixed))
        return sig + reserved + _u32(1) + struct.pack('>H', gamma_fixed)

    # Full curve LUT
    count = len(values)
    data = sig + reserved + _u32(count)
    for v in values:
        # Map 0.0-1.0 to 0-65535
        encoded = int(round(float(v) * 65535.0))
        encoded = max(0, min(65535, encoded))
        data += struct.pack('>H', encoded)
    return _pad4(data)


def _build_desc_type(text: str) -> bytes:
    """Build a profileDescriptionType tag element (ICC v2).

    Layout:
        'desc' (4) + reserved (4) + ASCII_length (4) + ASCII string (incl NUL)
        + unicode count (4) + unicode lang (4) + unicode data
        + scriptcode count (2) + scriptcode (1) + mac description (67)
    """
    sig = b'desc'
    reserved = b'\x00' * 4

    ascii_bytes = text.encode('ascii', errors='replace') + b'\x00'
    ascii_len = len(ascii_bytes)

    data = sig + reserved + _u32(ascii_len) + ascii_bytes

    # Unicode localizable description (empty)
    data += _u32(0)          # Unicode language code
    data += _u32(0)          # Unicode count

    # ScriptCode (Macintosh) description (empty)
    data += struct.pack('>H', 0)   # ScriptCode code
    data += b'\x00'                # ScriptCode count
    data += b'\x00' * 67          # Macintosh description (67 bytes)

    return _pad4(data)


def _build_text_type(text: str) -> bytes:
    """Build a textType tag element.

    Layout: 'text' (4) + reserved (4) + ASCII string (incl NUL).
    """
    sig = b'text'
    reserved = b'\x00' * 4
    ascii_bytes = text.encode('ascii', errors='replace') + b'\x00'
    return _pad4(sig + reserved + ascii_bytes)


# ---------------------------------------------------------------------------
# Profile header builder
# ---------------------------------------------------------------------------

def _build_header(
    profile_size: int,
    device_class: bytes = b'mntr',
    color_space: bytes = b'RGB ',
    pcs: bytes = b'XYZ ',
) -> bytes:
    """Build the 128-byte ICC v2 profile header.

    Parameters
    ----------
    profile_size : int
        Total file size in bytes (filled in after assembly).
    device_class : bytes
        4-byte device class signature (b'mntr', b'abst', etc.).
    color_space : bytes
        4-byte data color space (b'RGB ').
    pcs : bytes
        4-byte profile connection space (b'XYZ ').
    """
    now = datetime.datetime.utcnow()

    header = bytearray(128)

    # 0-3: Profile size (placeholder, will be patched)
    struct.pack_into('>I', header, 0, profile_size)

    # 4-7: Preferred CMM type (none)
    header[4:8] = b'\x00' * 4

    # 8-11: Profile version (2.4.0)
    struct.pack_into('>I', header, 8, ICC_VERSION_2_4)

    # 12-15: Device class
    header[12:16] = device_class

    # 16-19: Color space of data
    header[16:20] = color_space

    # 20-23: PCS
    header[20:24] = pcs

    # 24-35: Date and time (year, month, day, hour, minute, second as uint16)
    struct.pack_into('>HHHHHH', header, 24,
                     now.year, now.month, now.day,
                     now.hour, now.minute, now.second)

    # 36-39: 'acsp' signature (mandatory)
    header[36:40] = ICC_SIGNATURE

    # 40-43: Primary platform (Microsoft = 'MSFT')
    header[40:44] = b'MSFT'

    # 44-47: Profile flags
    struct.pack_into('>I', header, 44, 0)

    # 48-51: Device manufacturer
    header[48:52] = b'\x00' * 4

    # 52-55: Device model
    header[52:56] = b'\x00' * 4

    # 56-63: Device attributes (8 bytes)
    header[56:64] = b'\x00' * 8

    # 64-67: Rendering intent (0 = Perceptual)
    struct.pack_into('>I', header, 64, 0)

    # 68-79: PCS illuminant (D50 in s15Fixed16)
    struct.pack_into('>i', header, 68, int(round(D50_WHITE[0] * 65536)))
    struct.pack_into('>i', header, 72, int(round(D50_WHITE[1] * 65536)))
    struct.pack_into('>i', header, 76, int(round(D50_WHITE[2] * 65536)))

    # 80-83: Profile creator
    header[80:84] = b'DNGT'

    # 84-99: Profile ID (MD5) -- filled as zeros, computed later if desired
    header[84:100] = b'\x00' * 16

    # 100-127: Reserved
    header[100:128] = b'\x00' * 28

    return bytes(header)


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------

def _compute_profile_id(profile_data: bytearray) -> bytes:
    """Compute the ICC profile ID (MD5 of the profile with certain fields zeroed).

    Per ICC spec, fields at offsets 44-47 (flags), 64-67 (rendering intent),
    and 84-99 (profile ID) must be zeroed before computing the hash.
    """
    data = bytearray(profile_data)
    data[44:48] = b'\x00' * 4
    data[64:68] = b'\x00' * 4
    data[84:100] = b'\x00' * 16
    return hashlib.md5(bytes(data)).digest()


def _assemble_profile(tags: dict, device_class: bytes = b'mntr') -> bytes:
    """Assemble a complete ICC v2 profile from a dict of tag data.

    Parameters
    ----------
    tags : dict
        Mapping of 4-byte tag signature (bytes) to tag data (bytes).
    device_class : bytes
        ICC device class (b'mntr' or b'abst').

    Returns
    -------
    bytes
        Complete ICC profile data ready to be written to disk.
    """
    tag_count = len(tags)
    # Header = 128 bytes
    # Tag table = 4 (count) + 12 * tag_count
    tag_table_size = 4 + 12 * tag_count
    data_offset = 128 + tag_table_size

    # Pad data_offset to 4-byte boundary
    if data_offset % 4:
        data_offset += 4 - (data_offset % 4)

    # Build tag data block and directory
    tag_directory = []
    tag_data_block = bytearray()

    for sig, tag_bytes in tags.items():
        padded = _pad4(tag_bytes)
        offset = data_offset + len(tag_data_block)
        size = len(tag_bytes)
        tag_directory.append((sig, offset, size))
        tag_data_block += padded

    total_size = data_offset + len(tag_data_block)

    # Build header with correct size
    header = bytearray(_build_header(total_size, device_class=device_class))

    # Build tag table
    tag_table = _u32(tag_count)
    for sig, offset, size in tag_directory:
        tag_table += sig + _u32(offset) + _u32(size)

    # Assemble
    profile = bytearray(header)
    profile += tag_table
    # Pad between tag table end and data start
    padding_needed = data_offset - len(profile)
    if padding_needed > 0:
        profile += b'\x00' * padding_needed

    profile += tag_data_block

    # Patch profile size in header
    struct.pack_into('>I', profile, 0, len(profile))

    # Compute and insert profile ID (MD5)
    profile_id = _compute_profile_id(profile)
    profile[84:100] = profile_id

    return bytes(profile)


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------

def _trc_to_values(trc, num_entries: int = 256) -> Union[float, List[float]]:
    """Normalize a TRC argument to either a gamma float or a list of floats.

    Parameters
    ----------
    trc : float | int | list
        - float / int: gamma exponent (returned as-is).
        - list of float (0.0-1.0): returned as-is.
        - list of int (0-255): converted to 0.0-1.0 range.
        - list of (x, y) tuples: interpolated to *num_entries* uniform samples.
    """
    if isinstance(trc, (int, float)):
        return float(trc)

    if not isinstance(trc, (list, np.ndarray)):
        raise TypeError(f"TRC must be a number or list, got {type(trc).__name__}")

    trc = list(trc)
    if len(trc) == 0:
        return 1.0  # Identity gamma

    # Check if entries are tuples/lists (curve control points)
    first = trc[0]
    if isinstance(first, (tuple, list, np.ndarray)):
        # Interpolate (x, y) pairs to uniform 256-entry LUT
        points = sorted(trc, key=lambda p: p[0])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        # Determine input range
        x_max = max(xs) if max(xs) > 1.0 else 1.0
        y_max = max(ys) if max(ys) > 1.0 else 1.0
        # Normalize to 0.0-1.0
        if x_max > 1.0:
            xs = [x / 255.0 for x in xs]
        if y_max > 1.0:
            ys = [y / 255.0 for y in ys]
        result = list(np.interp(
            np.linspace(0.0, 1.0, num_entries),
            xs, ys
        ))
        return [max(0.0, min(1.0, v)) for v in result]

    # Plain list of values
    values = [float(v) for v in trc]
    # If values appear to be in 0-255 range, normalize
    if any(v > 1.0 for v in values):
        values = [v / 255.0 for v in values]
    return [max(0.0, min(1.0, v)) for v in values]


def _identity_curve_values(num_entries: int = 256) -> List[float]:
    """Generate an identity (linear) curve as a list of floats."""
    return [i / (num_entries - 1) for i in range(num_entries)]


def _gamma_curve_values(gamma: float, num_entries: int = 256) -> List[float]:
    """Generate a gamma curve as a list of floats."""
    return [(i / (num_entries - 1)) ** gamma for i in range(num_entries)]


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _matrix_to_colorants(matrix_3x3: np.ndarray) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """Extract column vectors from a 3x3 matrix as (rXYZ, gXYZ, bXYZ).

    The ICC matrix/TRC model defines:
        PCS_XYZ = [rXYZ | gXYZ | bXYZ] @ TRC(device_RGB)

    So each column of the matrix is a colorant XYZ triplet.

    Parameters
    ----------
    matrix_3x3 : np.ndarray
        3x3 matrix whose columns are the R, G, B colorant XYZ values.
    """
    m = np.asarray(matrix_3x3, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {m.shape}")
    r_xyz = (float(m[0, 0]), float(m[1, 0]), float(m[2, 0]))
    g_xyz = (float(m[0, 1]), float(m[1, 1]), float(m[2, 1]))
    b_xyz = (float(m[0, 2]), float(m[1, 2]), float(m[2, 2]))
    return r_xyz, g_xyz, b_xyz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_icc_profile(
    filepath: str,
    matrix_3x3: np.ndarray,
    trc_r=None,
    trc_g=None,
    trc_b=None,
    description: str = "DNG Channel Tool Profile",
    white_point: Tuple[float, float, float] = D50_WHITE,
    copyright_text: str = "Generated by DNG Channel Tool",
) -> str:
    """Write an ICC v2 Display (matrix/TRC) profile to disk.

    Parameters
    ----------
    filepath : str
        Destination path (.icc or .icm).
    matrix_3x3 : numpy.ndarray
        3x3 color matrix whose columns are the R, G, B colorant XYZ
        values.  For a channel-swap this would be the sRGB-to-XYZ matrix
        with permuted columns.
    trc_r, trc_g, trc_b : float | list | None
        Tone Response Curve for each channel.  Accepts:
        - ``None``: identity curve.
        - float: gamma exponent (e.g. 2.2).
        - list of 256 floats (0.0-1.0): explicit LUT.
        - list of (input, output) tuples: control-point curve.
    description : str
        Human-readable profile name stored in the 'desc' tag.
    white_point : tuple of float
        Media white point XYZ (default D50).
    copyright_text : str
        Copyright string stored in the 'cprt' tag.

    Returns
    -------
    str
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If *matrix_3x3* is not a valid 3x3 array.
    OSError
        If the file cannot be written.
    """
    matrix_3x3 = np.asarray(matrix_3x3, dtype=float)
    if matrix_3x3.shape != (3, 3):
        raise ValueError(f"matrix_3x3 must be shape (3,3), got {matrix_3x3.shape}")

    # --- Colorants from matrix columns ---
    r_xyz, g_xyz, b_xyz = _matrix_to_colorants(matrix_3x3)

    # --- TRC data ---
    trc_r_data = _trc_to_values(trc_r) if trc_r is not None else None
    trc_g_data = _trc_to_values(trc_g) if trc_g is not None else None
    trc_b_data = _trc_to_values(trc_b) if trc_b is not None else None

    # --- Build tags (ordered per convention) ---
    tags = {}
    tags[b'desc'] = _build_desc_type(description)
    tags[b'wtpt'] = _build_xyz_type(*white_point)
    tags[b'rXYZ'] = _build_xyz_type(*r_xyz)
    tags[b'gXYZ'] = _build_xyz_type(*g_xyz)
    tags[b'bXYZ'] = _build_xyz_type(*b_xyz)
    tags[b'rTRC'] = _build_curve_type(trc_r_data)
    tags[b'gTRC'] = _build_curve_type(trc_g_data)
    tags[b'bTRC'] = _build_curve_type(trc_b_data)
    tags[b'cprt'] = _build_text_type(copyright_text)

    profile_data = _assemble_profile(tags, device_class=b'mntr')

    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(profile_data)

    return filepath


def channel_swap_to_icc(
    filepath: str,
    mix_matrix: np.ndarray,
    description: str = "Channel Swap",
    gamma: float = 2.2,
) -> str:
    """Generate an ICC profile encoding a channel swap / mix.

    The mix_matrix is applied to the sRGB-D50 colorant matrix to produce
    a new set of colorants.  A standard sRGB gamma TRC is used unless
    overridden.

    Parameters
    ----------
    filepath : str
        Destination path (.icc or .icm).
    mix_matrix : numpy.ndarray
        3x3 channel mixing matrix where ``output = mix_matrix @ input``
        (same convention as :class:`channel_swap.MixMatrix`).
    description : str
        Profile description string.
    gamma : float
        Gamma exponent for all three TRC channels (default 2.2).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    mix_matrix = np.asarray(mix_matrix, dtype=float)
    if mix_matrix.shape != (3, 3):
        raise ValueError(f"mix_matrix must be shape (3,3), got {mix_matrix.shape}")

    # Apply the channel swap to the sRGB colorant matrix.
    # ICC model: PCS = M @ TRC(RGB)
    # A channel swap permutes the input channels, so effectively we permute
    # the columns of the colorant matrix: new_M = sRGB_M @ mix_matrix^T
    # (because mix_matrix maps input RGB to output RGB, and the columns of
    # the ICC matrix represent how each input channel contributes to XYZ).
    #
    # More precisely, if the swap says out_R = in_B, out_G = in_G, out_B = in_R
    # then the device's "R" channel should use the XYZ values of the original B
    # column, etc.  This means we rearrange columns of SRGB_MATRIX_D50 by
    # multiplying on the right by mix_matrix transposed.
    icc_matrix = SRGB_MATRIX_D50 @ mix_matrix.T

    return write_icc_profile(
        filepath=filepath,
        matrix_3x3=icc_matrix,
        trc_r=gamma,
        trc_g=gamma,
        trc_b=gamma,
        description=description,
    )


def tone_curve_to_icc(
    filepath: str,
    curve_points,
    description: str = "Tone Curve",
    per_channel: bool = False,
    curve_points_r=None,
    curve_points_g=None,
    curve_points_b=None,
) -> str:
    """Generate an ICC profile encoding tone curves with an identity matrix.

    Parameters
    ----------
    filepath : str
        Destination path (.icc or .icm).
    curve_points : list
        Master curve as a list of (input, output) tuples (0-255 or 0.0-1.0),
        a list of 256 values, or a single gamma float.
        Applied to all channels unless *per_channel* is True and individual
        channel curves are provided.
    description : str
        Profile description string.
    per_channel : bool
        If True, use *curve_points_r/g/b* for individual channels.
    curve_points_r, curve_points_g, curve_points_b : list, optional
        Per-channel curve overrides (same format as *curve_points*).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    if per_channel:
        trc_r = curve_points_r if curve_points_r is not None else curve_points
        trc_g = curve_points_g if curve_points_g is not None else curve_points
        trc_b = curve_points_b if curve_points_b is not None else curve_points
    else:
        trc_r = trc_g = trc_b = curve_points

    return write_icc_profile(
        filepath=filepath,
        matrix_3x3=SRGB_MATRIX_D50,
        trc_r=trc_r,
        trc_g=trc_g,
        trc_b=trc_b,
        description=description,
    )


def style_to_icc(
    filepath: str,
    image_style,
    description: str = "",
) -> str:
    """Generate an ICC profile from an :class:`style_transfer.ImageStyle` object.

    Extracts per-channel tone curves from the style and encodes them into a
    matrix/TRC ICC profile using sRGB colorants.

    Parameters
    ----------
    filepath : str
        Destination path (.icc or .icm).
    image_style : style_transfer.ImageStyle
        The style object containing tone curves and parameters.
    description : str
        Profile description.  If empty, uses ``image_style.name``.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    if not description:
        description = getattr(image_style, 'name', 'Imported Style') or 'Imported Style'

    # Extract per-channel curves (list of (x, y) tuples with 0-255 range)
    trc_r = getattr(image_style, 'tone_curve_r', None) or None
    trc_g = getattr(image_style, 'tone_curve_g', None) or None
    trc_b = getattr(image_style, 'tone_curve_b', None) or None

    # Fall back to master tone curve applied to all channels
    master = getattr(image_style, 'tone_curve', None) or None
    if trc_r is None and master is not None:
        trc_r = master
    if trc_g is None and master is not None:
        trc_g = master
    if trc_b is None and master is not None:
        trc_b = master

    # For monochrome styles, apply the luminance curve equally
    if getattr(image_style, 'is_monochrome', False) and master is not None:
        trc_r = trc_g = trc_b = master

    return write_icc_profile(
        filepath=filepath,
        matrix_3x3=SRGB_MATRIX_D50,
        trc_r=trc_r,
        trc_g=trc_g,
        trc_b=trc_b,
        description=description,
    )


# ---------------------------------------------------------------------------
# Utility: read back basic ICC info (for verification)
# ---------------------------------------------------------------------------

def read_icc_info(filepath: str) -> dict:
    """Read basic information from an ICC profile file.

    Returns a dict with keys: size, version, device_class, color_space,
    pcs, tag_count, tags (list of signature strings).

    Parameters
    ----------
    filepath : str
        Path to the .icc / .icm file.

    Returns
    -------
    dict
        Basic profile metadata.

    Raises
    ------
    ValueError
        If the file is not a valid ICC profile.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 132:
        raise ValueError("File too small to be a valid ICC profile")

    size = struct.unpack_from('>I', data, 0)[0]
    if size != len(data):
        raise ValueError(
            f"Profile size mismatch: header says {size}, file is {len(data)} bytes"
        )

    signature = data[36:40]
    if signature != ICC_SIGNATURE:
        raise ValueError(
            f"Invalid ICC signature: expected 'acsp', got {signature!r}"
        )

    version_raw = struct.unpack_from('>I', data, 8)[0]
    version_major = (version_raw >> 24) & 0xFF
    version_minor = (version_raw >> 20) & 0x0F

    device_class = data[12:16].decode('ascii', errors='replace').strip()
    color_space = data[16:20].decode('ascii', errors='replace').strip()
    pcs = data[20:24].decode('ascii', errors='replace').strip()

    tag_count = struct.unpack_from('>I', data, 128)[0]
    tags = []
    for i in range(tag_count):
        offset = 132 + i * 12
        tag_sig = data[offset:offset + 4].decode('ascii', errors='replace')
        tags.append(tag_sig)

    return {
        'size': size,
        'version': f"{version_major}.{version_minor}",
        'device_class': device_class,
        'color_space': color_space,
        'pcs': pcs,
        'tag_count': tag_count,
        'tags': tags,
    }
