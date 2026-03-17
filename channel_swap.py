"""
Color Channel Swap & Mix Logic

Supports:
- All 6 RGB permutations
- Weighted channel mixing (3x3 matrix)
- IR photography presets
- DCP color matrix / forward matrix transformation
- HueSatMap hue remapping for channel swaps
"""

import struct
import colorsys
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


# All 6 permutations of RGB
PERMUTATIONS = {
    "RGB (Original)": (0, 1, 2),
    "RBG (G↔B)":     (0, 2, 1),
    "GRB (R↔G)":     (1, 0, 2),
    "GBR":            (1, 2, 0),
    "BRG":            (2, 0, 1),
    "BGR (R↔B)":     (2, 1, 0),
}

# Common swap presets
SWAP_PRESETS = {
    "Original":         (0, 1, 2),
    "R ↔ G":            (1, 0, 2),
    "R ↔ B":            (2, 1, 0),
    "G ↔ B":            (0, 2, 1),
    "R → G → B → R":   (2, 0, 1),
    "R → B → G → R":   (1, 2, 0),
}

# IR-specific presets (name, 3x3 mix matrix)
IR_PRESETS = {
    "IR Standard (R<>B)": np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]),
    "IR Falschfarben": np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]),
    "IR Goldton": np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]),
    "IR Blauer Himmel": np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.0, 0.7],
    ]),
    "IR Monochrom": np.array([
        [0.33, 0.33, 0.34],
        [0.33, 0.33, 0.34],
        [0.33, 0.33, 0.34],
    ]),
}

CHANNEL_NAMES = ["R (Rot)", "G (Grün)", "B (Blau)"]
CHANNEL_SHORT = ["R", "G", "B"]


@dataclass
class ChannelMapping:
    """Defines which input channel maps to each output channel (permutation)."""
    r_source: int = 0  # 0=R, 1=G, 2=B
    g_source: int = 1
    b_source: int = 2

    @property
    def permutation(self) -> Tuple[int, int, int]:
        return (self.r_source, self.g_source, self.b_source)

    @property
    def name(self) -> str:
        return f"{CHANNEL_SHORT[self.r_source]}{CHANNEL_SHORT[self.g_source]}{CHANNEL_SHORT[self.b_source]}"

    @property
    def is_identity(self) -> bool:
        return self.r_source == 0 and self.g_source == 1 and self.b_source == 2

    def permutation_matrix(self) -> np.ndarray:
        """3x3 permutation matrix P: output = P @ input."""
        P = np.zeros((3, 3))
        P[0, self.r_source] = 1.0
        P[1, self.g_source] = 1.0
        P[2, self.b_source] = 1.0
        return P

    def to_mix_matrix(self) -> 'MixMatrix':
        """Convert to a MixMatrix."""
        return MixMatrix(matrix=self.permutation_matrix())

    @classmethod
    def from_permutation(cls, perm: Tuple[int, int, int]) -> 'ChannelMapping':
        return cls(r_source=perm[0], g_source=perm[1], b_source=perm[2])


@dataclass
class MixMatrix:
    """
    Weighted 3x3 channel mixing matrix.
    matrix[out][in] = weight of input channel 'in' contributing to output channel 'out'.
    """
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def is_identity(self) -> bool:
        return np.allclose(self.matrix, np.eye(3))

    @property
    def is_permutation(self) -> bool:
        """Check if this is a pure permutation (no mixing)."""
        m = self.matrix
        return (np.all((m == 0) | (m == 1)) and
                np.all(m.sum(axis=1) == 1) and
                np.all(m.sum(axis=0) == 1))

    @property
    def name(self) -> str:
        if self.is_permutation:
            perm = self.matrix.argmax(axis=1)
            return f"{CHANNEL_SHORT[perm[0]]}{CHANNEL_SHORT[perm[1]]}{CHANNEL_SHORT[perm[2]]}"
        return "Mix"

    def to_channel_mapping(self) -> Optional[ChannelMapping]:
        """Convert to ChannelMapping if this is a pure permutation."""
        if not self.is_permutation:
            return None
        perm = self.matrix.argmax(axis=1)
        return ChannelMapping(r_source=int(perm[0]), g_source=int(perm[1]), b_source=int(perm[2]))

    def normalize_rows(self) -> 'MixMatrix':
        """Normalize each row to sum to 1.0."""
        m = self.matrix.copy()
        for i in range(3):
            s = m[i].sum()
            if s > 0:
                m[i] /= s
        return MixMatrix(matrix=m)


# ── Image channel operations ──────────────────────────────────

def swap_image_channels(image: np.ndarray, mapping: ChannelMapping) -> np.ndarray:
    """Swap color channels in an RGB image (permutation only)."""
    if mapping.is_identity:
        return image.copy()
    result = np.empty_like(image)
    result[:, :, 0] = image[:, :, mapping.r_source]
    result[:, :, 1] = image[:, :, mapping.g_source]
    result[:, :, 2] = image[:, :, mapping.b_source]
    return result


def mix_image_channels(image: np.ndarray, mix: MixMatrix) -> np.ndarray:
    """Apply weighted channel mixing to an RGB image."""
    if mix.is_identity:
        return image.copy()

    # Work in float to avoid overflow
    img_f = image.astype(np.float32)
    m = mix.matrix.astype(np.float32)

    result = np.empty_like(img_f)
    for out_ch in range(3):
        result[:, :, out_ch] = (
            m[out_ch, 0] * img_f[:, :, 0] +
            m[out_ch, 1] * img_f[:, :, 1] +
            m[out_ch, 2] * img_f[:, :, 2]
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_to_image(image: np.ndarray, mix: MixMatrix) -> np.ndarray:
    """Apply mix matrix to image (uses fast path for permutations)."""
    mapping = mix.to_channel_mapping()
    if mapping is not None:
        return swap_image_channels(image, mapping)
    return mix_image_channels(image, mix)


# ── DCP matrix operations ────────────────────────────────────

def swap_color_matrix(matrix: np.ndarray, mapping: ChannelMapping) -> np.ndarray:
    """Apply channel swap to ColorMatrix (XYZ → CameraRGB): rearrange rows."""
    if mapping.is_identity or matrix is None:
        return matrix.copy() if matrix is not None else None
    result = np.empty_like(matrix)
    result[0, :] = matrix[mapping.r_source, :]
    result[1, :] = matrix[mapping.g_source, :]
    result[2, :] = matrix[mapping.b_source, :]
    return result


def swap_forward_matrix(matrix: np.ndarray, mapping: ChannelMapping) -> np.ndarray:
    """Apply channel swap to ForwardMatrix (CameraRGB → XYZ): rearrange columns."""
    if mapping.is_identity or matrix is None:
        return matrix.copy() if matrix is not None else None
    result = np.empty_like(matrix)
    result[:, 0] = matrix[:, mapping.r_source]
    result[:, 1] = matrix[:, mapping.g_source]
    result[:, 2] = matrix[:, mapping.b_source]
    return result


def mix_color_matrix(matrix: np.ndarray, mix: MixMatrix) -> Optional[np.ndarray]:
    """Apply mix matrix to ColorMatrix: new_CM = M @ CM."""
    if matrix is None:
        return None
    if mix.is_identity:
        return matrix.copy()
    return mix.matrix @ matrix


def mix_forward_matrix(matrix: np.ndarray, mix: MixMatrix) -> Optional[np.ndarray]:
    """Apply mix matrix to ForwardMatrix: new_FM = FM @ M_inv."""
    if matrix is None:
        return None
    if mix.is_identity:
        return matrix.copy()
    try:
        m_inv = np.linalg.inv(mix.matrix)
        return matrix @ m_inv
    except np.linalg.LinAlgError:
        return matrix.copy()


def apply_to_color_matrix(matrix: np.ndarray, mix: MixMatrix) -> Optional[np.ndarray]:
    """Apply mix to ColorMatrix (fast path for permutations)."""
    mapping = mix.to_channel_mapping()
    if mapping is not None:
        return swap_color_matrix(matrix, mapping)
    return mix_color_matrix(matrix, mix)


def apply_to_forward_matrix(matrix: np.ndarray, mix: MixMatrix) -> Optional[np.ndarray]:
    """Apply mix to ForwardMatrix (fast path for permutations)."""
    mapping = mix.to_channel_mapping()
    if mapping is not None:
        return swap_forward_matrix(matrix, mapping)
    return mix_forward_matrix(matrix, mix)


# ── HueSatMap remapping ───────────────────────────────────────

def remap_hue_sat_map(data: bytes, dims: tuple, mix: MixMatrix, endian: str = '<') -> bytes:
    """
    Remap HueSatMap data for a channel swap/mix.

    The HueSatMap is indexed by (hue, saturation, value).
    Each entry has 3 floats: (hue_shift, sat_scale, val_scale).

    For channel permutations, we remap hue bins by calculating
    how the swap rotates the hue wheel.
    """
    num_hues, num_sats, num_vals = dims
    num_entries = num_hues * num_sats * num_vals
    expected_size = num_entries * 3 * 4  # 3 floats * 4 bytes

    if len(data) < expected_size:
        return data  # Can't process, return unchanged

    # Parse float data
    floats = struct.unpack(f'{endian}{num_entries * 3}f', data[:expected_size])
    arr = np.array(floats).reshape(num_vals, num_sats, num_hues, 3)
    result = np.zeros_like(arr)

    m = mix.matrix

    # Build hue remapping table
    hue_remap = {}
    for h_idx in range(num_hues):
        hue_angle = (h_idx + 0.5) / num_hues  # 0..1

        # Create RGB at this hue (full sat, mid value)
        r, g, b = colorsys.hsv_to_rgb(hue_angle, 1.0, 1.0)

        # Apply mix
        rgb = np.array([r, g, b])
        new_rgb = np.clip(m @ rgb, 0, 1)

        # Convert back to HSV
        new_h, new_s, new_v = colorsys.rgb_to_hsv(
            float(new_rgb[0]), float(new_rgb[1]), float(new_rgb[2])
        )

        new_h_idx = int(new_h * num_hues) % num_hues
        hue_remap[h_idx] = new_h_idx

    # Remap entries
    for h_idx, new_h_idx in hue_remap.items():
        for v_idx in range(num_vals):
            for s_idx in range(num_sats):
                result[v_idx, s_idx, new_h_idx] = arr[v_idx, s_idx, h_idx]

    # Pack back
    return struct.pack(f'{endian}{num_entries * 3}f', *result.flatten().tolist())
