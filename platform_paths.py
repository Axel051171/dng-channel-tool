"""
Plattformübergreifende Pfade und Systemfunktionen

Zentrales Modul für alle OS-abhängigen Operationen:
- Adobe Camera Raw / Lightroom Verzeichnisse
- SD-Karten-Erkennung
- Datei-Explorer öffnen
- High-DPI Konfiguration
"""

import os
import sys
import logging
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Plattform-Erkennung ──────────────────────────────────

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')


# ── Adobe-Verzeichnisse ─────────────────────────────────

def get_adobe_profile_dir() -> str:
    """Gibt das Adobe CameraRaw Camera-Profiles Verzeichnis zurück.

    Plattform-spezifische Pfade:
        Windows: %APPDATA%/Adobe/CameraRaw/CameraProfiles
        macOS:   ~/Library/Application Support/Adobe/CameraRaw/CameraProfiles
        Linux:   ~/.config/Adobe/CameraRaw/CameraProfiles
    """
    base = _get_adobe_base_dir()
    return os.path.join(base, 'CameraRaw', 'CameraProfiles')


def get_lightroom_preset_dir() -> Path:
    """Gibt das Adobe Camera Raw Settings-Verzeichnis zurück.

    Plattform-spezifische Pfade:
        Windows: %APPDATA%/Adobe/CameraRaw/Settings
        macOS:   ~/Library/Application Support/Adobe/CameraRaw/Settings
        Linux:   ~/.config/Adobe/CameraRaw/Settings
    """
    base = _get_adobe_base_dir()
    return Path(base) / 'CameraRaw' / 'Settings'


def _get_adobe_base_dir() -> str:
    """Basis-Verzeichnis für Adobe-Produkte je nach Plattform."""
    if IS_WINDOWS:
        appdata = os.environ.get('APPDATA', '')
        if appdata:
            return os.path.join(appdata, 'Adobe')
        return os.path.join(os.path.expanduser('~'),
                            'AppData', 'Roaming', 'Adobe')

    if IS_MACOS:
        return os.path.join(os.path.expanduser('~'),
                            'Library', 'Application Support', 'Adobe')

    # Linux / andere
    return os.path.join(os.path.expanduser('~'), '.config', 'Adobe')


# ── SD-Karten-Erkennung ─────────────────────────────────

def find_sd_cards() -> List[str]:
    """Findet SD-Karten/Wechseldatenträger plattformübergreifend.

    Sucht nach Laufwerken mit NIKON/DCIM Verzeichnissen.
    """
    if IS_WINDOWS:
        return _find_sd_cards_windows()
    elif IS_MACOS:
        return _find_sd_cards_macos()
    else:
        return _find_sd_cards_linux()


def _find_sd_cards_windows() -> List[str]:
    """Windows: Laufwerksbuchstaben durchsuchen."""
    import string
    cards = []
    for letter in string.ascii_uppercase[3:]:  # D: onwards
        drive = f"{letter}:\\"
        if not os.path.exists(drive):
            continue
        if _is_camera_card(drive):
            cards.append(drive)
        else:
            try:
                import ctypes
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
                if drive_type == 2:  # DRIVE_REMOVABLE
                    cards.append(drive)
            except (AttributeError, OSError) as e:
                logger.debug("Laufwerkstyp-Prüfung fehlgeschlagen für %s: %s", drive, e)
    return cards


def _find_sd_cards_macos() -> List[str]:
    """macOS: /Volumes durchsuchen."""
    cards = []
    volumes_dir = '/Volumes'
    if not os.path.isdir(volumes_dir):
        return cards
    for name in os.listdir(volumes_dir):
        mount = os.path.join(volumes_dir, name)
        if os.path.isdir(mount) and name != 'Macintosh HD':
            if _is_camera_card(mount):
                cards.append(mount)
    return cards


def _find_sd_cards_linux() -> List[str]:
    """Linux: /media/$USER und /run/media/$USER durchsuchen."""
    cards = []
    user = os.environ.get('USER', os.environ.get('LOGNAME', ''))
    search_dirs = [
        f'/media/{user}',
        f'/run/media/{user}',
        '/mnt',
    ]
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            mount = os.path.join(base, name)
            if os.path.isdir(mount) and _is_camera_card(mount):
                cards.append(mount)
    return cards


def _is_camera_card(path: str) -> bool:
    """Prüft ob ein Laufwerk wie eine Kamera-SD-Karte aussieht."""
    return (os.path.isdir(os.path.join(path, 'NIKON')) or
            os.path.isdir(os.path.join(path, 'DCIM')) or
            os.path.isdir(os.path.join(path, 'CANONMSC')) or
            os.path.isdir(os.path.join(path, 'PRIVATE', 'SONY')))


# ── Datei-Explorer öffnen ────────────────────────────────

def open_in_file_manager(path: str):
    """Öffnet einen Ordner im System-Dateimanager.

    Windows: explorer.exe
    macOS: open
    Linux: xdg-open
    """
    if not os.path.exists(path):
        logger.warning("Pfad existiert nicht: %s", path)
        return

    folder = path if os.path.isdir(path) else os.path.dirname(path)

    try:
        if IS_WINDOWS:
            os.startfile(folder)
        elif IS_MACOS:
            subprocess.Popen(['open', folder])
        else:
            subprocess.Popen(['xdg-open', folder])
    except Exception as e:
        logger.warning("Dateimanager konnte nicht geöffnet werden: %s", e)


# ── High-DPI Konfiguration ──────────────────────────────

def setup_high_dpi():
    """Aktiviert High-DPI Awareness für die aktuelle Plattform.

    Muss VOR dem Erstellen des Tk-Root-Fensters aufgerufen werden.
    """
    if IS_WINDOWS:
        try:
            import ctypes
            # Per-Monitor DPI Awareness v2 (Windows 10 1703+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try:
                # Fallback: System DPI Awareness
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass

    # macOS und Linux: Tk skaliert meist automatisch
    # Für Tk unter Linux kann TK_SCALING gesetzt werden
    if IS_LINUX:
        # Versuche GDK_SCALE zu lesen
        gdk_scale = os.environ.get('GDK_SCALE', '')
        if gdk_scale:
            try:
                os.environ.setdefault('TK_SCALING', str(float(gdk_scale)))
            except ValueError:
                pass
