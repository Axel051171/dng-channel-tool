"""
Preset-Bibliothek mit Thumbnails (#4)

Scannt alle installierten Presets (Adobe XMP, DCP, Nikon NPC/NP3)
und zeigt sie in einer durchsuchbaren Übersicht mit Vorschaubildern.
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PresetEntry:
    """Ein Eintrag in der Preset-Bibliothek."""
    name: str
    filepath: str
    format: str          # "XMP", "DCP", "NPC", "NP3", "CUBE"
    category: str = ""   # Ordnername / Gruppe
    camera: str = ""     # Kamera-Modell (falls bekannt)
    description: str = ""
    is_monochrome: bool = False


def scan_adobe_presets() -> List[PresetEntry]:
    """Scannt alle Adobe Camera Raw Presets."""
    entries = []
    appdata = os.environ.get('APPDATA', '')
    if not appdata:
        return entries

    # XMP Presets (Settings)
    settings_dir = os.path.join(appdata, 'Adobe', 'CameraRaw', 'Settings')
    if os.path.isdir(settings_dir):
        for root, dirs, files in os.walk(settings_dir):
            for f in files:
                if f.lower().endswith('.xmp'):
                    rel = os.path.relpath(root, settings_dir)
                    category = rel if rel != '.' else "Allgemein"
                    entries.append(PresetEntry(
                        name=os.path.splitext(f)[0],
                        filepath=os.path.join(root, f),
                        format="XMP",
                        category=category,
                    ))

    # DCP Profiles (CameraProfiles)
    profiles_dir = os.path.join(appdata, 'Adobe', 'CameraRaw', 'CameraProfiles')
    if os.path.isdir(profiles_dir):
        for root, dirs, files in os.walk(profiles_dir):
            for f in files:
                if f.lower().endswith('.dcp'):
                    rel = os.path.relpath(root, profiles_dir)
                    category = rel if rel != '.' else "Kamera-Profile"
                    name = os.path.splitext(f)[0]
                    entries.append(PresetEntry(
                        name=name,
                        filepath=os.path.join(root, f),
                        format="DCP",
                        category=category,
                        camera=_extract_camera_from_name(name),
                    ))

    return entries


def scan_nikon_presets() -> List[PresetEntry]:
    """Scannt nach Nikon NPC/NP3 Dateien in bekannten Ordnern."""
    entries = []
    search_dirs = [
        os.path.expanduser('~/Documents'),
        os.path.expanduser('~/Downloads'),
    ]

    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for ext in ('*.npc', '*.NPC', '*.np3', '*.NP3', '*.ncp', '*.NCP'):
            for filepath in glob.glob(os.path.join(base_dir, '**', ext), recursive=True):
                name = os.path.splitext(os.path.basename(filepath))[0]
                fmt = os.path.splitext(filepath)[1].upper().lstrip('.')
                entries.append(PresetEntry(
                    name=name,
                    filepath=filepath,
                    format=fmt,
                    category="Nikon",
                ))

    return entries


def scan_lut_files() -> List[PresetEntry]:
    """Scannt nach .cube LUT-Dateien."""
    entries = []
    search_dirs = [
        os.path.expanduser('~/Documents'),
        os.path.expanduser('~/Downloads'),
    ]

    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for filepath in glob.glob(os.path.join(base_dir, '**', '*.cube'), recursive=True):
            name = os.path.splitext(os.path.basename(filepath))[0]
            entries.append(PresetEntry(
                name=name,
                filepath=filepath,
                format="CUBE",
                category="LUTs",
            ))

    return entries


def scan_all_presets() -> List[PresetEntry]:
    """Scannt alle bekannten Preset-Quellen."""
    all_entries = []
    all_entries.extend(scan_adobe_presets())
    all_entries.extend(scan_nikon_presets())
    all_entries.extend(scan_lut_files())

    # Sort by category then name
    all_entries.sort(key=lambda e: (e.format, e.category, e.name))
    return all_entries


def filter_presets(entries: List[PresetEntry], query: str = "",
                    format_filter: str = "") -> List[PresetEntry]:
    """Filtert Presets nach Suchbegriff und Format."""
    result = entries

    if query:
        q = query.lower()
        result = [e for e in result if
                  q in e.name.lower() or
                  q in e.category.lower() or
                  q in e.camera.lower()]

    if format_filter:
        result = [e for e in result if e.format == format_filter]

    return result


def get_preset_info(entry: PresetEntry) -> dict:
    """Liest Detail-Informationen aus einer Preset-Datei."""
    info = {
        'name': entry.name,
        'format': entry.format,
        'path': entry.filepath,
        'size': os.path.getsize(entry.filepath),
        'category': entry.category,
    }

    try:
        if entry.format == "XMP":
            info.update(_parse_xmp_info(entry.filepath))
        elif entry.format == "DCP":
            info.update(_parse_dcp_info(entry.filepath))
        elif entry.format in ("NPC", "NP3", "NCP"):
            info.update(_parse_npc_info(entry.filepath))
    except Exception:
        pass

    return info


def _parse_xmp_info(filepath: str) -> dict:
    """Liest grundlegende Infos aus einer XMP-Datei."""
    info = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        profile = re.search(r'crs:CameraProfile="([^"]*)"', content)
        if profile:
            info['camera_profile'] = profile.group(1)

        grayscale = re.search(r'ConvertToGrayscale="([^"]*)"', content)
        if grayscale:
            info['monochrome'] = grayscale.group(1) == 'True'

        wb = re.search(r'crs:WhiteBalance="([^"]*)"', content)
        if wb:
            info['white_balance'] = wb.group(1)
    except Exception:
        pass
    return info


def _parse_dcp_info(filepath: str) -> dict:
    """Liest Infos aus einer DCP-Datei."""
    info = {}
    try:
        from dcp_io import DCPReader
        profile = DCPReader().read(filepath)
        info['camera_model'] = profile.camera_model
        info['profile_name'] = profile.profile_name
        info['dual_illuminant'] = profile.has_dual_illuminant()
    except Exception:
        pass
    return info


def _parse_npc_info(filepath: str) -> dict:
    """Liest Infos aus einer NPC/NP3-Datei."""
    info = {}
    try:
        from npc_io import read_npc
        pc = read_npc(filepath)
        info['pc_name'] = pc.name
        info['pc_base'] = pc.base
        info['tone_curve_points'] = len(pc.tone_curve)
    except Exception:
        pass
    return info


def _extract_camera_from_name(name: str) -> str:
    """Versucht den Kameranamen aus dem Dateinamen zu extrahieren."""
    # Common patterns: "Canon EOS 5D Mark IV ...", "Sony ILCE-7M3 ..."
    brands = ['Canon', 'Nikon', 'Sony', 'Fujifilm', 'Panasonic',
              'Olympus', 'Pentax', 'Leica', 'Hasselblad', 'Apple', 'Samsung']
    for brand in brands:
        if brand.lower() in name.lower():
            return name.split(' Infrared')[0].split(' Chrome')[0].strip()
    return ""
