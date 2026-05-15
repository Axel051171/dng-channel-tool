"""
Camera Database - loads color matrices from dnglab TOML files.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera information with color matrices."""
    make: str
    model: str
    clean_make: str
    clean_model: str
    color_matrix_a: Optional[np.ndarray] = None   # Illuminant A
    color_matrix_d65: Optional[np.ndarray] = None  # Illuminant D65
    toml_path: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.clean_make} {self.clean_model}"

    @property
    def unique_camera_model(self) -> str:
        """Model string as used in DNG/DCP files."""
        return self.model


def _parse_toml_simple(filepath: str) -> dict:
    """
    Minimal TOML parser for the dnglab camera files.
    Only handles the fields we need (strings, arrays of floats).
    """
    result = {}
    current_section = result

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Section header
            if line.startswith('['):
                section_name = line.strip('[]').strip()
                if section_name == 'cameras.color_matrix':
                    if 'color_matrix' not in result:
                        result['color_matrix'] = {}
                    current_section = result['color_matrix']
                else:
                    current_section = result
                continue

            # Key = value
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()

                # String value
                if value.startswith('"'):
                    value = value.strip('"')
                    current_section[key] = value
                # Array of numbers
                elif value.startswith('['):
                    # Extract numbers from array
                    nums_str = value.strip('[]')
                    try:
                        nums = [float(x.strip()) for x in nums_str.split(',') if x.strip()]
                        current_section[key] = nums
                    except ValueError:
                        current_section[key] = value
                # Number
                elif value.replace('.', '').replace('-', '').isdigit():
                    try:
                        current_section[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        current_section[key] = value
                else:
                    current_section[key] = value

    return result


_camera_cache: Dict[str, Tuple[float, List[CameraInfo]]] = {}


def _max_toml_mtime(cameras_dir: str) -> float:
    """Maximale mtime über cameras_dir und alle enthaltenen .toml-Dateien.

    Erfasst auch INHALTLICHE Änderungen einzelner TOML-Dateien (das
    Verzeichnis-mtime allein bleibt unverändert wenn nur ein File-Inhalt
    aktualisiert wird).
    """
    try:
        max_t = os.path.getmtime(cameras_dir)
    except OSError:
        return 0.0
    for brand_dir in os.listdir(cameras_dir):
        brand_path = os.path.join(cameras_dir, brand_dir)
        if not os.path.isdir(brand_path):
            continue
        try:
            max_t = max(max_t, os.path.getmtime(brand_path))
            for f in os.listdir(brand_path):
                if f.endswith('.toml'):
                    try:
                        max_t = max(max_t, os.path.getmtime(
                            os.path.join(brand_path, f)))
                    except OSError:
                        pass
        except OSError:
            continue
    return max_t


def invalidate_camera_cache(dnglab_path: Optional[str] = None) -> None:
    """Cache leeren (z.B. nach Hot-Reload). Ohne Argument: kompletter Cache."""
    if dnglab_path is None:
        _camera_cache.clear()
        return
    cameras_dir = os.path.join(dnglab_path, 'rawler', 'data', 'cameras')
    _camera_cache.pop(cameras_dir, None)


def load_camera_database(dnglab_path: str) -> List[CameraInfo]:
    """
    Load camera database from dnglab's rawler/data/cameras/ directory.
    Verwendet einen Cache: Nur neu geladen wenn sich Dateien geändert haben.

    Args:
        dnglab_path: Path to dnglab repository root

    Returns:
        List of CameraInfo objects
    """
    cameras_dir = os.path.join(dnglab_path, 'rawler', 'data', 'cameras')
    cameras = []

    if not os.path.isdir(cameras_dir):
        return cameras

    # Cache prüfen: maximale mtime über alle TOML-Dateien (deckt auch
    # Inhaltsänderungen ab, nicht nur Add/Remove)
    dir_mtime = _max_toml_mtime(cameras_dir)
    if cameras_dir in _camera_cache:
        cached_mtime, cached_cameras = _camera_cache[cameras_dir]
        if cached_mtime >= dir_mtime > 0:
            return cached_cameras

    for brand_dir in sorted(os.listdir(cameras_dir)):
        brand_path = os.path.join(cameras_dir, brand_dir)
        if not os.path.isdir(brand_path):
            continue

        for toml_file in sorted(os.listdir(brand_path)):
            if not toml_file.endswith('.toml'):
                continue

            filepath = os.path.join(brand_path, toml_file)
            try:
                data = _parse_toml_simple(filepath)

                make = data.get('make', brand_dir)
                model = data.get('model', toml_file.replace('.toml', ''))
                clean_make = data.get('clean_make', make)
                clean_model = data.get('clean_model', model)

                cam = CameraInfo(
                    make=make,
                    model=model,
                    clean_make=clean_make,
                    clean_model=clean_model,
                    toml_path=filepath,
                )

                # Parse color matrices
                cm = data.get('color_matrix', {})
                if 'A' in cm and len(cm['A']) == 9:
                    cam.color_matrix_a = np.array(cm['A']).reshape(3, 3)
                if 'D65' in cm and len(cm['D65']) == 9:
                    cam.color_matrix_d65 = np.array(cm['D65']).reshape(3, 3)

                if cam.color_matrix_a is not None or cam.color_matrix_d65 is not None:
                    cameras.append(cam)

            except Exception as e:
                logger.debug(
                    "Kamera-Datenbank: Eintrag übersprungen: %s", e)
                continue

    # Cache mit der VOR dem Scan gemessenen mtime aktualisieren (vermeidet
    # Race wenn während des Scans Dateien geändert wurden).
    _camera_cache[cameras_dir] = (dir_mtime, cameras)

    return cameras


def find_dnglab_path() -> Optional[str]:
    """Try to find dnglab repository relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check sibling directory
    parent = os.path.dirname(script_dir)
    candidate = os.path.join(parent, 'dnglab')
    if os.path.isdir(os.path.join(candidate, 'rawler', 'data', 'cameras')):
        return candidate
    return None
