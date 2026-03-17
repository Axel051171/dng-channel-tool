"""
Canon Picture Style & Sony Creative Look Konverter

Liest und schreibt Canon Picture Style Dateien (.pf2/.pf3) und
Sony Creative Look Presets (XML). Konvertiert zwischen allen
Kamera-Herstellerformaten und exportiert als Adobe Lightroom XMP-Presets
oder Nikon Picture Controls.

Canon PF3:
- Binaerformat mit "DSTO"-Header (EOS R Serie)
- Parameter: Sharpness, Contrast, Saturation, Color Tone, Tonkurve
- PF2 ist das aeltere Format (EOS 5D III, 7D II etc.)

Sony Creative Look:
- XML-basiertes Format fuer die Kamera-Looks
- Parameter: Contrast, Highlights, Shadows, Fade, Saturation, Sharpness, Clarity
"""

import struct
import uuid
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from xml.etree import ElementTree as ET


# =====================================================================
#  Canon Picture Style
# =====================================================================

CANON_BASE_STYLES = [
    "Standard", "Portrait", "Landscape", "Neutral", "Faithful",
    "Monochrome", "Fine Detail", "User Def 1", "User Def 2", "User Def 3",
]

CANON_BASE_STYLE_IDS = {name: idx for idx, name in enumerate(CANON_BASE_STYLES)}

CANON_FILTER_EFFECTS = {
    0: None,
    1: "Yellow",
    2: "Orange",
    3: "Red",
    4: "Green",
}
CANON_FILTER_BY_NAME = {v: k for k, v in CANON_FILTER_EFFECTS.items() if v}

CANON_TONING_EFFECTS = {
    0: None,
    1: "Sepia",
    2: "Blue",
    3: "Purple",
    4: "Green",
}
CANON_TONING_BY_NAME = {v: k for k, v in CANON_TONING_EFFECTS.items() if v}

# Magic-Bytes bekannter PF-Formate
PF3_MAGIC = b"DSTO"
PF2_MAGIC = b"DSTP"


@dataclass
class CanonPictureStyle:
    """Repraesentiert einen Canon Picture Style (.pf2/.pf3).

    Wertebereiche:
        sharpness:    0-7  (aeltere Kameras) oder 0-10 (EOS R Serie)
        fineness:     1-5  (nur neuere Kameras, Feinheit der Schaerfung)
        threshold:    1-5  (nur neuere Kameras, Schwellenwert der Schaerfung)
        contrast:     -4 bis +4
        saturation:   -4 bis +4
        color_tone:   -4 bis +4
        filter_effect: nur bei Monochrome
        toning_effect: nur bei Monochrome
        tone_curve:   Liste von (input, output) Kontrollpunkten (0-255)
    """
    name: str = "Custom"
    base_style: str = "Standard"

    # Schaerfung
    sharpness: int = 3
    fineness: int = 3
    threshold: int = 2

    # Tonwert-Einstellungen (-4 bis +4)
    contrast: int = 0
    saturation: int = 0
    color_tone: int = 0

    # Monochrom-Optionen
    filter_effect: Optional[str] = None   # None/Yellow/Orange/Red/Green
    toning_effect: Optional[str] = None   # None/Sepia/Blue/Purple/Green

    # Tonkurve als Kontrollpunkte (input, output) im Bereich 0-255
    tone_curve: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 0), (255, 255)]
    )

    @property
    def is_monochrome(self) -> bool:
        """Prueft ob der Stil ein Monochrom-Stil ist."""
        return self.base_style.lower() == "monochrome"

    @property
    def base_style_id(self) -> int:
        """Gibt die numerische ID des Basisstils zurueck."""
        return CANON_BASE_STYLE_IDS.get(self.base_style, 0)


# ── Canon PF3 Reader ─────────────────────────────────────

def read_canon_pf3(filepath: str) -> CanonPictureStyle:
    """Liest eine Canon Picture Style Datei (.pf2/.pf3).

    Versucht den DSTO-Header (PF3) und den DSTP-Header (PF2) zu erkennen.
    Die Binaerstruktur ist nicht vollstaendig dokumentiert; wir extrahieren
    die bekannten Parameter so gut wie moeglich.

    Args:
        filepath: Pfad zur .pf2 oder .pf3 Datei

    Returns:
        CanonPictureStyle mit den extrahierten Parametern

    Raises:
        ValueError: Wenn die Datei kein gueltiger Canon Picture Style ist
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 32:
        raise ValueError(f"Datei zu kurz fuer Canon Picture Style: {len(data)} Bytes")

    magic = data[:4]
    if magic not in (PF3_MAGIC, PF2_MAGIC):
        raise ValueError(
            f"Unbekanntes Format: Magic {magic!r} "
            f"(erwartet {PF3_MAGIC!r} oder {PF2_MAGIC!r})"
        )

    style = CanonPictureStyle()
    style.name = os.path.splitext(os.path.basename(filepath))[0]

    is_pf3 = (magic == PF3_MAGIC)

    # ── Header parsen ──
    # Bytes 4-7: Versionsinfo / Flags
    # Bytes 8-11: Datenoffset oder Groesse
    # Die genaue Struktur variiert, daher suchen wir nach Mustern

    if is_pf3:
        _parse_pf3_body(data, style)
    else:
        _parse_pf2_body(data, style)

    return style


def _parse_pf3_body(data: bytes, style: CanonPictureStyle):
    """Internes Parsing des PF3-Formats (EOS R Serie)."""
    # PF3 hat typischerweise den Stilnamen ab Offset ~16-48
    # und Parameter in einem Tagged-Block danach

    # Versuche den Stilnamen zu finden (null-terminierter ASCII-String)
    name = _extract_ascii_string(data, 16, 32)
    if name:
        style.name = name

    # Basis-Stil erkennen (suche nach bekannten Strings)
    data_str = data.decode('ascii', errors='replace').lower()
    for bs in CANON_BASE_STYLES:
        if bs.lower() in data_str:
            style.base_style = bs
            break

    # Parameter-Block suchen
    # Canon kodiert Parameter oft als signierte Bytes im Bereich -4..+4
    # In der Binaerdatei als offset-Werte (0x80 + Wert) oder direkt
    _extract_pf3_params(data, style)

    # Tonkurve suchen (Kontrollpunkte-Block)
    _extract_canon_tone_curve(data, style)


def _parse_pf2_body(data: bytes, style: CanonPictureStyle):
    """Internes Parsing des PF2-Formats (aeltere EOS Kameras)."""
    name = _extract_ascii_string(data, 16, 32)
    if name:
        style.name = name

    data_str = data.decode('ascii', errors='replace').lower()
    for bs in CANON_BASE_STYLES:
        if bs.lower() in data_str:
            style.base_style = bs
            break

    _extract_pf2_params(data, style)
    _extract_canon_tone_curve(data, style)


def _extract_ascii_string(data: bytes, start: int, max_len: int) -> str:
    """Extrahiert einen null-terminierten ASCII-String."""
    end = start + max_len
    if end > len(data):
        end = len(data)

    result = []
    for i in range(start, end):
        b = data[i]
        if b == 0:
            break
        if 0x20 <= b <= 0x7E:
            result.append(chr(b))
        else:
            break
    return "".join(result).strip()


def _extract_pf3_params(data: bytes, style: CanonPictureStyle):
    """Extrahiert Parameter aus PF3-Daten.

    Die Parameter liegen typischerweise in einem Block ab Offset ~48-80.
    Format: Tag (1 Byte) + Wert (1 Byte, signed oder unsigned).
    """
    if len(data) < 60:
        return

    # Heuristik: Suche nach einem Bereich mit vielen kleinen Werten
    # die zu den bekannten Parameterbereichen passen
    # Typischer Parameter-Block: Sharpness(0-10), Contrast(-4..+4), etc.

    # Bekannte Offsets fuer verschiedene PF3-Versionen
    for param_offset in (48, 52, 56, 60, 64):
        if param_offset + 8 > len(data):
            continue

        candidate = data[param_offset:param_offset + 8]

        # Sharpness: 0-10, Fineness: 1-5, Threshold: 1-5
        sharp = candidate[0]
        fine = candidate[1]
        thresh = candidate[2]

        if 0 <= sharp <= 10 and 1 <= fine <= 5 and 1 <= thresh <= 5:
            style.sharpness = sharp
            style.fineness = fine
            style.threshold = thresh

            # Contrast, Saturation, Color Tone als signed bytes
            # Gespeichert als Offset von 0x80 oder direkt als signed
            for val_offset in (3, 4, 5):
                raw = candidate[val_offset]
                if raw > 127:
                    # Interpretiere als signed byte
                    val = raw - 256
                else:
                    val = raw

                # Auf gueltigen Bereich clippen
                val = max(-4, min(4, val))

                if val_offset == 3:
                    style.contrast = val
                elif val_offset == 4:
                    style.saturation = val
                elif val_offset == 5:
                    style.color_tone = val
            break

    # Monochrom-Optionen
    if style.is_monochrome:
        # Filter- und Tonung-Effekt suchen
        for i in range(48, min(len(data) - 2, 120)):
            filt = data[i]
            tone = data[i + 1]
            if filt in CANON_FILTER_EFFECTS and tone in CANON_TONING_EFFECTS:
                if filt > 0 or tone > 0:
                    style.filter_effect = CANON_FILTER_EFFECTS.get(filt)
                    style.toning_effect = CANON_TONING_EFFECTS.get(tone)
                    break


def _extract_pf2_params(data: bytes, style: CanonPictureStyle):
    """Extrahiert Parameter aus PF2-Daten (aelteres Format).

    Aeltere Kameras haben nur Sharpness 0-7 (kein Fineness/Threshold).
    """
    if len(data) < 52:
        return

    for param_offset in (40, 44, 48, 52):
        if param_offset + 6 > len(data):
            continue

        candidate = data[param_offset:param_offset + 6]
        sharp = candidate[0]

        if 0 <= sharp <= 7:
            style.sharpness = sharp
            style.fineness = 3  # Default
            style.threshold = 2  # Default

            for val_offset in (1, 2, 3):
                raw = candidate[val_offset]
                val = raw - 256 if raw > 127 else raw
                val = max(-4, min(4, val))

                if val_offset == 1:
                    style.contrast = val
                elif val_offset == 2:
                    style.saturation = val
                elif val_offset == 3:
                    style.color_tone = val
            break


def _extract_canon_tone_curve(data: bytes, style: CanonPictureStyle):
    """Sucht und extrahiert eine Tonkurve aus Canon PF-Daten.

    Canon speichert Tonkurven als Folge von (X, Y)-Kontrollpunkten,
    wobei die Anzahl der Punkte vorher steht.
    """
    # Suche nach einem Tonkurven-Block:
    # Format: num_points (1-2 Bytes), dann num_points * 2 Bytes (X, Y Paare)
    # Wir suchen nach einem plausiblen Muster
    for offset in range(64, len(data) - 20):
        num_pts = data[offset]
        if 2 <= num_pts <= 20:
            # Pruefe ob die folgenden Bytes plausible Kurvenpunkte sind
            valid = True
            points = []
            prev_x = -1

            for i in range(num_pts):
                idx = offset + 1 + i * 2
                if idx + 1 >= len(data):
                    valid = False
                    break
                x, y = data[idx], data[idx + 1]
                # X-Werte sollten monoton steigend sein
                if x <= prev_x:
                    valid = False
                    break
                # Erster Punkt nahe (0,0), letzter nahe (255,255) erwartet
                prev_x = x
                points.append((x, y))

            if valid and len(points) >= 2:
                # Plausibilitaetscheck: erster Punkt nahe 0, letzter nahe 255
                if points[0][0] <= 10 and points[-1][0] >= 240:
                    style.tone_curve = points
                    return


# ── Canon PF3 Writer ─────────────────────────────────────

def write_canon_pf3(filepath: str, style: CanonPictureStyle):
    """Schreibt eine Canon Picture Style Datei (.pf3).

    Hinweis: Das PF3-Format ist nicht vollstaendig dokumentiert.
    Die erzeugte Datei enthaelt alle Parameter in einem lesbaren Format,
    ist aber moeglicherweise nicht direkt von der Kamera importierbar.
    Fuer den Transfer zwischen Systemen sollten die Konvertierungsfunktionen
    (canon_to_lightroom_xmp, canon_to_nikon_npc) bevorzugt werden.

    Args:
        filepath: Ausgabepfad (sollte auf .pf3 enden)
        style: Der zu schreibende Picture Style
    """
    out = bytearray()

    # ── Header ──
    out += PF3_MAGIC                       # Magic "DSTO"
    out += struct.pack('>I', 0x00010000)   # Version 1.0
    out += struct.pack('>I', 64)           # Datenoffset
    out += struct.pack('>I', 0)            # Reserved

    # ── Name (32 Bytes, null-terminiert) ──
    name_bytes = style.name[:31].encode('ascii', errors='replace')
    out += name_bytes + b'\x00' * (32 - len(name_bytes))

    # ── Basis-Stil (4 Bytes) ──
    out += struct.pack('>I', style.base_style_id)

    # ── Padding bis Offset 64 ──
    while len(out) < 64:
        out += b'\x00'

    # ── Parameter-Block ──
    # Sharpness, Fineness, Threshold (unsigned)
    out += bytes([
        max(0, min(10, style.sharpness)),
        max(1, min(5, style.fineness)),
        max(1, min(5, style.threshold)),
    ])

    # Contrast, Saturation, Color Tone (signed, als Byte)
    for val in (style.contrast, style.saturation, style.color_tone):
        clamped = max(-4, min(4, val))
        out += struct.pack('b', clamped)

    # ── Monochrom-Optionen ──
    filter_id = CANON_FILTER_BY_NAME.get(style.filter_effect, 0)
    toning_id = CANON_TONING_BY_NAME.get(style.toning_effect, 0)
    out += bytes([filter_id, toning_id])

    # ── Padding ──
    out += b'\x00' * 4

    # ── Tonkurve ──
    curve = style.tone_curve or [(0, 0), (255, 255)]
    num_pts = min(len(curve), 20)
    out += bytes([num_pts])
    for i in range(num_pts):
        x, y = curve[i]
        out += bytes([max(0, min(255, x)), max(0, min(255, y))])

    # Restliche Slots auffuellen
    for _ in range(num_pts, 20):
        out += b'\x00\x00'

    # ── Ende-Marker ──
    out += b'\x00' * 16

    with open(filepath, 'wb') as f:
        f.write(out)


# ── Canon → Lightroom XMP ────────────────────────────────

def canon_to_lightroom_xmp(style: CanonPictureStyle, filepath: str):
    """Konvertiert einen Canon Picture Style zu einem Adobe Lightroom XMP-Preset.

    Mapping:
        Canon Contrast (-4..+4)    → LR Contrast (-100..+100)
        Canon Saturation (-4..+4)  → LR Saturation (-100..+100)
        Canon Color Tone (-4..+4)  → LR Tint Shift
        Canon Sharpness (0-10)     → LR Sharpness (0-150)
        Canon Tone Curve           → LR ToneCurvePV2012

    Args:
        style: Canon Picture Style
        filepath: Ausgabepfad fuer die .xmp Datei
    """
    preset_uuid = str(uuid.uuid4()).upper()

    # Parameter-Mapping
    lr_contrast = int(style.contrast * 25)          # -100..+100
    lr_saturation = int(style.saturation * 25)      # -100..+100
    lr_tint = int(style.color_tone * 5)             # Tint-Verschiebung
    lr_sharpness = max(0, int(style.sharpness * 15))  # 0-150
    lr_clarity = 0

    grayscale = "True" if style.is_monochrome else "False"

    # Tonkurve als XMP
    curve_items = []
    for x, y in style.tone_curve:
        curve_items.append(f'    <rdf:li>{x}, {y}</rdf:li>')
    curve_xml = "\n".join(curve_items) if curve_items else \
        '    <rdf:li>0, 0</rdf:li>\n    <rdf:li>255, 255</rdf:li>'

    xmp = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - Canon Picture Style">
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
    crs:Copyright="DNG Channel Tool - Canon Picture Style Converter"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:Contrast="{lr_contrast}"
    crs:Saturation="{lr_saturation}"
    crs:Tint="{lr_tint}"
    crs:Sharpness="{lr_sharpness}"
    crs:Clarity="{lr_clarity}"
    crs:ConvertToGrayscale="{grayscale}"
    crs:ToneCurveName2012="Custom"
    crs:Group{{Name}}="Canon Picture Styles"
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

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp)


# ── Canon → Nikon PC ─────────────────────────────────────

def canon_to_nikon_npc(style: CanonPictureStyle):
    """Konvertiert einen Canon Picture Style zu Nikon Picture Control Parametern.

    Mapping:
        Canon Base Style → Nikon Base Profile
        Canon Contrast (-4..+4) → Nikon Contrast (0-255, Mitte 128)
        Canon Saturation (-4..+4) → Nikon Saturation (0-255, Mitte 128)
        Canon Sharpness (0-10) → Nikon Sharpening (0-255)
        Canon Tone Curve → Nikon Tone Curve

    Returns:
        NikonPictureControlFile
    """
    from npc_io import NikonPictureControlFile

    pc = NikonPictureControlFile()
    pc.name = style.name[:19]

    # Basis-Stil Mapping: Canon → Nikon
    _canon_to_nikon_base = {
        "Standard": "STANDARD",
        "Portrait": "PORTRAIT",
        "Landscape": "LANDSCAPE",
        "Neutral": "NEUTRAL",
        "Faithful": "FLAT",
        "Monochrome": "MONOCHROME",
        "Fine Detail": "STANDARD",
        "User Def 1": "STANDARD",
        "User Def 2": "STANDARD",
        "User Def 3": "STANDARD",
    }
    pc.base = _canon_to_nikon_base.get(style.base_style, "STANDARD")

    # Canon (-4..+4) → Nikon unsigned (0-255, Mitte=128)
    # Skalierung: 1 Canon-Stufe ≈ 16 Nikon-Einheiten
    pc.contrast = max(0, min(255, 128 + style.contrast * 16))
    pc.brightness = 128  # Canon hat keinen direkten Brightness-Parameter
    pc.saturation = max(0, min(255, 128 + style.saturation * 16))
    pc.hue = max(0, min(255, 128 + style.color_tone * 8))

    # Sharpness: Canon 0-10 → Nikon 0-255
    pc.sharpening = max(0, min(255, int(style.sharpness * 25.5)))

    # Monochrom-Optionen
    if style.is_monochrome:
        # Filter-Effekt Mapping
        _filter_map = {
            "Yellow": 1, "Orange": 2, "Red": 3, "Green": 4,
        }
        pc.filter_effect = _filter_map.get(style.filter_effect, 0)

        # Toning-Effekt Mapping
        _toning_map = {
            "Sepia": 2, "Blue": 8, "Purple": 9, "Green": 6,
        }
        pc.toning_effect = _toning_map.get(style.toning_effect, 0)

    # Tonkurve uebernehmen
    if style.tone_curve:
        pc.tone_curve = list(style.tone_curve)

    return pc


# =====================================================================
#  Sony Creative Look
# =====================================================================

SONY_LOOK_NAMES = {
    "ST": "Standard",
    "PT": "Portrait",
    "NT": "Neutral",
    "VV": "Vivid",
    "VV2": "Vivid 2",
    "FL": "Film",
    "IN": "Instant",
    "SH": "Soft High-key",
    "BW": "B&W",
    "SE": "Sepia",
}


@dataclass
class SonyCreativeLook:
    """Repraesentiert einen Sony Creative Look.

    Parameter-Bereiche (wie in der Kamera):
        contrast:        -9 bis +9
        highlights:      -9 bis +9
        shadows:         -9 bis +9
        fade:             0 bis +9
        saturation:      -9 bis +9
        sharpness:       -9 bis +9
        sharpness_range: -9 bis +9
        clarity:         -9 bis +9
    """
    name: str = "ST"
    display_name: str = "Standard"

    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    fade: int = 0
    saturation: int = 0
    sharpness: int = 0
    sharpness_range: int = 0
    clarity: int = 0

    @property
    def is_monochrome(self) -> bool:
        """Prueft ob es ein Schwarzweiss-Look ist."""
        return self.name.upper() in ("BW", "SE")


def create_sony_look(
    name: str,
    contrast: int = 0,
    highlights: int = 0,
    shadows: int = 0,
    fade: int = 0,
    saturation: int = 0,
    sharpness: int = 0,
    clarity: int = 0,
    sharpness_range: int = 0,
) -> SonyCreativeLook:
    """Erstellt einen neuen Sony Creative Look mit den angegebenen Parametern.

    Args:
        name: Kurzname des Looks (z.B. "ST", "VV", "FL")
        contrast:        -9 bis +9
        highlights:      -9 bis +9
        shadows:         -9 bis +9
        fade:             0 bis +9
        saturation:      -9 bis +9
        sharpness:       -9 bis +9
        clarity:         -9 bis +9
        sharpness_range: -9 bis +9

    Returns:
        SonyCreativeLook
    """
    clamp9 = lambda v: max(-9, min(9, v))

    return SonyCreativeLook(
        name=name.upper(),
        display_name=SONY_LOOK_NAMES.get(name.upper(), name),
        contrast=clamp9(contrast),
        highlights=clamp9(highlights),
        shadows=clamp9(shadows),
        fade=max(0, min(9, fade)),
        saturation=clamp9(saturation),
        sharpness=clamp9(sharpness),
        sharpness_range=clamp9(sharpness_range),
        clarity=clamp9(clarity),
    )


# ── Vordefinierte Sony Basis-Looks ───────────────────────

SONY_BASE_LOOKS: Dict[str, SonyCreativeLook] = {
    "ST": SonyCreativeLook(
        name="ST", display_name="Standard",
        contrast=0, highlights=0, shadows=0, fade=0,
        saturation=0, sharpness=3, sharpness_range=0, clarity=0,
    ),
    "PT": SonyCreativeLook(
        name="PT", display_name="Portrait",
        contrast=0, highlights=0, shadows=1, fade=0,
        saturation=-1, sharpness=2, sharpness_range=0, clarity=0,
    ),
    "NT": SonyCreativeLook(
        name="NT", display_name="Neutral",
        contrast=-1, highlights=0, shadows=0, fade=0,
        saturation=-1, sharpness=2, sharpness_range=0, clarity=0,
    ),
    "VV": SonyCreativeLook(
        name="VV", display_name="Vivid",
        contrast=1, highlights=0, shadows=0, fade=0,
        saturation=3, sharpness=3, sharpness_range=0, clarity=1,
    ),
    "VV2": SonyCreativeLook(
        name="VV2", display_name="Vivid 2",
        contrast=1, highlights=0, shadows=0, fade=0,
        saturation=4, sharpness=3, sharpness_range=0, clarity=2,
    ),
    "FL": SonyCreativeLook(
        name="FL", display_name="Film",
        contrast=1, highlights=-2, shadows=2, fade=2,
        saturation=0, sharpness=2, sharpness_range=0, clarity=0,
    ),
    "IN": SonyCreativeLook(
        name="IN", display_name="Instant",
        contrast=2, highlights=-3, shadows=2, fade=3,
        saturation=-1, sharpness=2, sharpness_range=0, clarity=0,
    ),
    "SH": SonyCreativeLook(
        name="SH", display_name="Soft High-key",
        contrast=-2, highlights=2, shadows=-1, fade=1,
        saturation=-1, sharpness=1, sharpness_range=0, clarity=-2,
    ),
    "BW": SonyCreativeLook(
        name="BW", display_name="B&W",
        contrast=1, highlights=0, shadows=0, fade=0,
        saturation=0, sharpness=3, sharpness_range=0, clarity=1,
    ),
    "SE": SonyCreativeLook(
        name="SE", display_name="Sepia",
        contrast=1, highlights=0, shadows=1, fade=1,
        saturation=-2, sharpness=2, sharpness_range=0, clarity=0,
    ),
}


# ── Sony Creative Look XML I/O ───────────────────────────

def read_sony_look_xml(filepath: str) -> SonyCreativeLook:
    """Liest einen Sony Creative Look aus einer XML-Datei.

    Args:
        filepath: Pfad zur XML-Datei

    Returns:
        SonyCreativeLook

    Raises:
        ValueError: Wenn die Datei keinen gueltigen Creative Look enthaelt
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    look = SonyCreativeLook()

    # Name aus dem Dateinamen oder XML-Tag
    name_elem = root.find('.//Name')
    if name_elem is not None and name_elem.text:
        look.name = name_elem.text.strip()
        look.display_name = SONY_LOOK_NAMES.get(look.name, look.name)
    else:
        look.name = os.path.splitext(os.path.basename(filepath))[0]
        look.display_name = look.name

    # Parameter auslesen (verschiedene XML-Strukturen versuchen)
    _param_tags = {
        'Contrast': 'contrast',
        'Highlight': 'highlights',
        'Highlights': 'highlights',
        'Shadow': 'shadows',
        'Shadows': 'shadows',
        'Fade': 'fade',
        'Saturation': 'saturation',
        'Sharpness': 'sharpness',
        'SharpnessRange': 'sharpness_range',
        'Clarity': 'clarity',
    }

    for xml_tag, attr_name in _param_tags.items():
        elem = root.find(f'.//{xml_tag}')
        if elem is not None and elem.text:
            try:
                setattr(look, attr_name, int(elem.text.strip()))
            except ValueError:
                pass

    return look


def write_sony_look_xml(filepath: str, look: SonyCreativeLook):
    """Schreibt einen Sony Creative Look als XML-Datei.

    Args:
        filepath: Ausgabepfad
        look: Der Creative Look
    """
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreativeLook>
  <Name>{look.name}</Name>
  <DisplayName>{look.display_name}</DisplayName>
  <Contrast>{look.contrast}</Contrast>
  <Highlights>{look.highlights}</Highlights>
  <Shadows>{look.shadows}</Shadows>
  <Fade>{look.fade}</Fade>
  <Saturation>{look.saturation}</Saturation>
  <Sharpness>{look.sharpness}</Sharpness>
  <SharpnessRange>{look.sharpness_range}</SharpnessRange>
  <Clarity>{look.clarity}</Clarity>
  <IsMonochrome>{str(look.is_monochrome).lower()}</IsMonochrome>
</CreativeLook>
"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xml)


# ── Sony → Lightroom XMP ─────────────────────────────────

def sony_to_lightroom_xmp(look: SonyCreativeLook, filepath: str):
    """Konvertiert einen Sony Creative Look zu einem Adobe Lightroom XMP-Preset.

    Mapping:
        Sony Contrast (-9..+9)     → LR Contrast (-100..+100)
        Sony Highlights (-9..+9)   → LR Highlights2012 (-100..+100)
        Sony Shadows (-9..+9)      → LR Shadows2012 (-100..+100)
        Sony Fade (0..+9)          → LR ToneCurve (angehobene Schatten)
        Sony Saturation (-9..+9)   → LR Saturation (-100..+100)
        Sony Sharpness (-9..+9)    → LR Sharpness (0-150)
        Sony Clarity (-9..+9)      → LR Clarity (-100..+100)

    Args:
        look: Sony Creative Look
        filepath: Ausgabepfad fuer die .xmp Datei
    """
    preset_uuid = str(uuid.uuid4()).upper()

    # Parameter-Mapping (-9..+9 → Lightroom-Bereiche)
    lr_contrast = int(look.contrast * 100 / 9)
    lr_highlights = int(look.highlights * 100 / 9)
    lr_shadows = int(look.shadows * 100 / 9)
    lr_saturation = int(look.saturation * 100 / 9)
    lr_sharpness = max(0, int(40 + look.sharpness * 12))
    lr_clarity = int(look.clarity * 100 / 9)

    grayscale = "True" if look.is_monochrome else "False"

    # Fade → Tonkurve mit angehobenen Schwarzwerten
    fade_curve = ""
    if look.fade > 0:
        # Fade hebt den Schwarzpunkt an
        black_lift = int(look.fade * 255 / 36)  # max ~64 bei fade=9
        fade_curve = f"""
   <crs:ToneCurvePV2012>
    <rdf:Seq>
    <rdf:li>0, {black_lift}</rdf:li>
    <rdf:li>64, {64 + black_lift // 2}</rdf:li>
    <rdf:li>128, 128</rdf:li>
    <rdf:li>192, 192</rdf:li>
    <rdf:li>255, 255</rdf:li>
    </rdf:Seq>
   </crs:ToneCurvePV2012>"""
        tone_curve_name = '    crs:ToneCurveName2012="Custom"'
    else:
        tone_curve_name = ""

    xmp = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool - Sony Creative Look">
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
    crs:Copyright="DNG Channel Tool - Sony Creative Look Converter"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:Contrast="{lr_contrast}"
    crs:Highlights2012="{lr_highlights}"
    crs:Shadows2012="{lr_shadows}"
    crs:Saturation="{lr_saturation}"
    crs:Sharpness="{lr_sharpness}"
    crs:Clarity="{lr_clarity}"
    crs:ConvertToGrayscale="{grayscale}"
{tone_curve_name}
    crs:Group{{Name}}="Sony Creative Looks"
    >{fade_curve}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xmp)


# ── Sony → Nikon PC ──────────────────────────────────────

def sony_to_nikon_npc(look: SonyCreativeLook):
    """Konvertiert einen Sony Creative Look zu Nikon Picture Control Parametern.

    Mapping:
        Sony Base → Nikon Base Profile
        Sony Contrast (-9..+9)   → Nikon Contrast (0-255, Mitte 128)
        Sony Highlights/Shadows  → Nikon Contrast + Brightness kombiniert
        Sony Saturation (-9..+9) → Nikon Saturation (0-255, Mitte 128)
        Sony Sharpness (-9..+9)  → Nikon Sharpening (0-255)
        Sony Clarity (-9..+9)    → Nikon Clarity (0-255)

    Returns:
        NikonPictureControlFile
    """
    from npc_io import NikonPictureControlFile

    pc = NikonPictureControlFile()
    pc.name = (look.display_name or look.name)[:19]

    # Basis-Profil Mapping
    _sony_to_nikon_base = {
        "ST": "STANDARD",
        "PT": "PORTRAIT",
        "NT": "NEUTRAL",
        "VV": "VIVID",
        "VV2": "VIVID",
        "FL": "FLAT",
        "IN": "FLAT",
        "SH": "PORTRAIT",
        "BW": "MONOCHROME",
        "SE": "MONOCHROME",
    }
    pc.base = _sony_to_nikon_base.get(look.name.upper(), "STANDARD")

    # Sony (-9..+9) → Nikon (0-255, Mitte=128)
    # Skalierung: 1 Sony-Stufe ≈ 14 Nikon-Einheiten (128/9 ≈ 14.2)
    scale = 128 / 9

    # Contrast: Sony-Highlights und Shadows fliessen mit ein
    combined_contrast = look.contrast + (look.highlights - look.shadows) * 0.3
    pc.contrast = max(0, min(255, int(128 + combined_contrast * scale)))

    # Brightness: abgeleitet aus Shadows und Fade
    brightness_shift = look.shadows * 0.5 + look.fade * 0.3
    pc.brightness = max(0, min(255, int(128 + brightness_shift * scale)))

    # Saturation
    pc.saturation = max(0, min(255, int(128 + look.saturation * scale)))

    # Sharpness
    pc.sharpening = max(0, min(255, int(128 + look.sharpness * scale)))

    # Clarity
    pc.clarity = max(0, min(255, int(128 + look.clarity * scale)))

    # Monochrom: Sepia-Toning fuer SE
    if look.name.upper() == "SE":
        pc.toning_effect = 2  # Sepia

    # Fade → Tonkurve mit angehobenem Schwarzpunkt
    if look.fade > 0:
        black_lift = int(look.fade * 255 / 36)
        pc.tone_curve = [
            (0, black_lift),
            (64, 64 + black_lift // 2),
            (128, 128),
            (192, 192),
            (255, 255),
        ]

    return pc


# =====================================================================
#  Kreuz-Konvertierung: Nikon ↔ Canon ↔ Sony
# =====================================================================

def canon_to_nikon(style: CanonPictureStyle):
    """Alias fuer canon_to_nikon_npc (Konsistenz mit anderen Modulen).

    Returns:
        NikonPictureControlFile
    """
    return canon_to_nikon_npc(style)


def canon_to_sony(style: CanonPictureStyle) -> SonyCreativeLook:
    """Konvertiert einen Canon Picture Style zu einem Sony Creative Look.

    Mapping:
        Canon Contrast (-4..+4) → Sony Contrast (-9..+9)
        Canon Saturation (-4..+4) → Sony Saturation (-9..+9)
        Canon Color Tone (-4..+4) → beeinflusst Sony Highlights/Shadows
        Canon Sharpness (0-10) → Sony Sharpness (-9..+9)

    Args:
        style: Canon Picture Style

    Returns:
        SonyCreativeLook
    """
    # Canon (-4..+4) → Sony (-9..+9): Faktor ≈ 2.25
    canon_to_sony_scale = 9 / 4

    look = SonyCreativeLook()
    look.name = "ST"
    look.display_name = style.name

    # Basis-Zuordnung
    _base_map = {
        "Standard": "ST",
        "Portrait": "PT",
        "Landscape": "VV",
        "Neutral": "NT",
        "Faithful": "NT",
        "Monochrome": "BW",
        "Fine Detail": "ST",
    }
    base_key = _base_map.get(style.base_style, "ST")
    look.name = base_key

    look.contrast = max(-9, min(9, int(style.contrast * canon_to_sony_scale)))
    look.saturation = max(-9, min(9, int(style.saturation * canon_to_sony_scale)))

    # Color Tone beeinflusst Highlights/Shadows leicht
    look.highlights = max(-9, min(9, int(style.color_tone * 0.5)))
    look.shadows = 0

    # Sharpness: Canon 0-10 → Sony -9..+9 (5 = neutral → 0)
    look.sharpness = max(-9, min(9, int((style.sharpness - 5) * 1.8)))

    look.clarity = 0
    look.fade = 0

    # Monochrom-Anpassungen
    if style.is_monochrome:
        look.name = "BW"
        if style.toning_effect == "Sepia":
            look.name = "SE"

    look.display_name = f"{style.name} (Canon)"
    return look


def sony_to_canon(look: SonyCreativeLook) -> CanonPictureStyle:
    """Konvertiert einen Sony Creative Look zu einem Canon Picture Style.

    Mapping:
        Sony Contrast (-9..+9)   → Canon Contrast (-4..+4)
        Sony Saturation (-9..+9) → Canon Saturation (-4..+4)
        Sony Sharpness (-9..+9)  → Canon Sharpness (0-10)
        Sony Highlights/Shadows  → Canon Color Tone (approximiert)

    Args:
        look: Sony Creative Look

    Returns:
        CanonPictureStyle
    """
    # Sony (-9..+9) → Canon (-4..+4): Faktor ≈ 0.44
    sony_to_canon_scale = 4 / 9

    style = CanonPictureStyle()
    style.name = look.display_name or look.name

    # Basis-Zuordnung
    _base_map = {
        "ST": "Standard",
        "PT": "Portrait",
        "NT": "Neutral",
        "VV": "Landscape",
        "VV2": "Landscape",
        "FL": "Standard",
        "IN": "Standard",
        "SH": "Portrait",
        "BW": "Monochrome",
        "SE": "Monochrome",
    }
    style.base_style = _base_map.get(look.name.upper(), "Standard")

    style.contrast = max(-4, min(4, round(look.contrast * sony_to_canon_scale)))
    style.saturation = max(-4, min(4, round(look.saturation * sony_to_canon_scale)))

    # Highlights/Shadows-Differenz → Color Tone (grobe Approximation)
    style.color_tone = max(-4, min(4, round(
        (look.highlights - look.shadows) * sony_to_canon_scale * 0.5
    )))

    # Sharpness: Sony -9..+9 → Canon 0-10 (0 = neutral → 5)
    style.sharpness = max(0, min(10, int(5 + look.sharpness * 5 / 9)))
    style.fineness = 3
    style.threshold = 2

    # Monochrom-Optionen
    if look.name.upper() == "SE":
        style.toning_effect = "Sepia"
    elif look.name.upper() == "BW":
        style.filter_effect = None
        style.toning_effect = None

    return style


def nikon_to_canon(npc) -> CanonPictureStyle:
    """Konvertiert eine Nikon Picture Control zu einem Canon Picture Style.

    Mapping:
        Nikon Contrast (0-255, Mitte 128) → Canon Contrast (-4..+4)
        Nikon Saturation (0-255, Mitte 128) → Canon Saturation (-4..+4)
        Nikon Sharpening (0-255) → Canon Sharpness (0-10)
        Nikon Hue (0-255, Mitte 128) → Canon Color Tone (-4..+4)
        Nikon Tone Curve → Canon Tone Curve

    Args:
        npc: NikonPictureControlFile (aus npc_io)

    Returns:
        CanonPictureStyle
    """
    style = CanonPictureStyle()
    style.name = npc.name or "Nikon Import"

    # Basis-Stil Mapping: Nikon → Canon
    _nikon_to_canon_base = {
        "STANDARD": "Standard",
        "NEUTRAL": "Neutral",
        "VIVID": "Landscape",
        "MONOCHROME": "Monochrome",
        "PORTRAIT": "Portrait",
        "LANDSCAPE": "Landscape",
        "FLAT": "Faithful",
        "DREAM": "Standard",
        "MORNING": "Standard",
        "POP": "Landscape",
        "SUNDAY": "Standard",
        "SOMBER": "Neutral",
        "DRAMATIC": "Standard",
        "SILENCE": "Neutral",
        "BLEACHED": "Standard",
        "MELANCHOLIC": "Neutral",
        "PURE": "Standard",
        "DENIM": "Standard",
        "TOY": "Landscape",
        "SEPIA": "Monochrome",
        "BLUE": "Monochrome",
        "RED": "Monochrome",
        "PINK": "Monochrome",
        "CHARCOAL": "Monochrome",
        "GRAPHITE": "Monochrome",
        "BINARY": "Monochrome",
        "CARBON": "Monochrome",
        "RICH TONE": "Standard",
    }
    style.base_style = _nikon_to_canon_base.get(npc.base.upper(), "Standard")

    # Nikon unsigned (0-255, Mitte=128) → Canon (-4..+4)
    # 128 Nikon-Einheiten Differenz → 4 Canon-Stufen → 1 Canon-Stufe = 32 Nikon
    def nikon_to_canon_val(val, default=128):
        if val is None:
            return 0
        return max(-4, min(4, round((val - default) / 32)))

    style.contrast = nikon_to_canon_val(npc.contrast)
    style.saturation = nikon_to_canon_val(npc.saturation)
    style.color_tone = nikon_to_canon_val(npc.hue)

    # Sharpness: Nikon 0-255 → Canon 0-10
    if npc.sharpening is not None:
        style.sharpness = max(0, min(10, round(npc.sharpening * 10 / 255)))
    else:
        style.sharpness = 4  # Default: Auto → mittlerer Wert

    style.fineness = 3
    style.threshold = 2

    # Monochrom-Optionen
    if npc.is_monochrome:
        style.base_style = "Monochrome"

        # Filter-Mapping Nikon → Canon
        from npc_io import FILTER_EFFECTS
        nikon_filter = FILTER_EFFECTS.get(npc.filter_effect)
        _nikon_filter_to_canon = {
            "Yellow": "Yellow",
            "Orange": "Orange",
            "Red": "Red",
            "Green": "Green",
            "Yellow-Green": "Green",
        }
        style.filter_effect = _nikon_filter_to_canon.get(nikon_filter)

        # Toning-Mapping Nikon → Canon
        from npc_io import TONING_EFFECTS
        nikon_toning = TONING_EFFECTS.get(npc.toning_effect)
        _nikon_toning_to_canon = {
            "Sepia": "Sepia",
            "Blue": "Blue",
            "Blue-Green": "Blue",
            "Purple-Blue": "Purple",
            "Red-Purple": "Purple",
            "Green": "Green",
        }
        style.toning_effect = _nikon_toning_to_canon.get(nikon_toning)

    # Tonkurve uebernehmen
    if npc.tone_curve and len(npc.tone_curve) >= 2:
        style.tone_curve = list(npc.tone_curve)

    return style


def nikon_to_sony(npc) -> SonyCreativeLook:
    """Konvertiert eine Nikon Picture Control zu einem Sony Creative Look.

    Mapping:
        Nikon Base Profile → Sony Look Name
        Nikon Contrast (0-255, Mitte 128) → Sony Contrast (-9..+9)
        Nikon Saturation (0-255, Mitte 128) → Sony Saturation (-9..+9)
        Nikon Sharpening (0-255) → Sony Sharpness (-9..+9)
        Nikon Clarity (0-255, Mitte 128) → Sony Clarity (-9..+9)
        Nikon Brightness (0-255, Mitte 128) → Sony Shadows (-9..+9)

    Args:
        npc: NikonPictureControlFile (aus npc_io)

    Returns:
        SonyCreativeLook
    """
    look = SonyCreativeLook()
    look.display_name = npc.name or "Nikon Import"

    # Basis-Profil Mapping
    _nikon_to_sony_base = {
        "STANDARD": "ST",
        "NEUTRAL": "NT",
        "VIVID": "VV",
        "MONOCHROME": "BW",
        "PORTRAIT": "PT",
        "LANDSCAPE": "VV",
        "FLAT": "FL",
        "DREAM": "SH",
        "MORNING": "FL",
        "POP": "VV2",
        "SUNDAY": "SH",
        "SOMBER": "NT",
        "DRAMATIC": "IN",
        "SILENCE": "NT",
        "BLEACHED": "SH",
        "MELANCHOLIC": "FL",
        "PURE": "ST",
        "DENIM": "FL",
        "TOY": "VV2",
        "SEPIA": "SE",
        "BLUE": "BW",
        "RED": "BW",
        "PINK": "BW",
        "CHARCOAL": "BW",
        "GRAPHITE": "BW",
        "BINARY": "BW",
        "CARBON": "BW",
        "RICH TONE": "VV",
    }
    look.name = _nikon_to_sony_base.get(npc.base.upper(), "ST")

    # Nikon unsigned (0-255, Mitte=128) → Sony (-9..+9)
    # 128 Nikon-Differenz → 9 Sony-Stufen → 1 Sony-Stufe ≈ 14.2 Nikon
    scale = 9 / 128

    def nikon_to_sony_val(val, default=128):
        if val is None:
            return 0
        return max(-9, min(9, round((val - default) * scale)))

    look.contrast = nikon_to_sony_val(npc.contrast)
    look.saturation = nikon_to_sony_val(npc.saturation)

    # Brightness → Shadows-Einfluss
    look.shadows = nikon_to_sony_val(npc.brightness)
    look.highlights = 0

    # Sharpness: Nikon Center=128 → Sony -9..+9
    look.sharpness = nikon_to_sony_val(npc.sharpening)

    # Clarity
    look.clarity = nikon_to_sony_val(npc.clarity)

    # Fade aus Tonkurve ableiten (angehobener Schwarzpunkt)
    if npc.tone_curve and len(npc.tone_curve) >= 2:
        first_point = npc.tone_curve[0]
        if first_point[0] <= 5 and first_point[1] > 5:
            # Schwarzpunkt ist angehoben → Fade
            look.fade = max(0, min(9, int(first_point[1] * 9 / 64)))

    return look


def sony_to_nikon(look: SonyCreativeLook):
    """Alias fuer sony_to_nikon_npc (Konsistenz mit anderen Modulen).

    Returns:
        NikonPictureControlFile
    """
    return sony_to_nikon_npc(look)


# =====================================================================
#  Hilfsfunktionen
# =====================================================================

def _clamp(val: int, lo: int, hi: int) -> int:
    """Begrenzt einen Wert auf den angegebenen Bereich."""
    return max(lo, min(hi, val))


def format_canon_style(style: CanonPictureStyle) -> str:
    """Formatiert einen Canon Picture Style als lesbaren Text.

    Returns:
        Mehrzeiliger String mit allen Parametern
    """
    lines = [
        f"Canon Picture Style: {style.name}",
        f"  Basis:       {style.base_style}",
        f"  Schaerfung:  {style.sharpness}",
        f"  Feinheit:    {style.fineness}",
        f"  Schwelle:    {style.threshold}",
        f"  Kontrast:    {style.contrast:+d}",
        f"  Saettigung:  {style.saturation:+d}",
        f"  Farbton:     {style.color_tone:+d}",
    ]
    if style.is_monochrome:
        lines.append(f"  Filter:      {style.filter_effect or 'Aus'}")
        lines.append(f"  Tonung:      {style.toning_effect or 'Aus'}")
    if style.tone_curve and len(style.tone_curve) > 2:
        lines.append(f"  Tonkurve:    {len(style.tone_curve)} Punkte")
    return "\n".join(lines)


def format_sony_look(look: SonyCreativeLook) -> str:
    """Formatiert einen Sony Creative Look als lesbaren Text.

    Returns:
        Mehrzeiliger String mit allen Parametern
    """
    lines = [
        f"Sony Creative Look: {look.display_name} ({look.name})",
        f"  Kontrast:        {look.contrast:+d}",
        f"  Lichter:         {look.highlights:+d}",
        f"  Schatten:        {look.shadows:+d}",
        f"  Verblassen:      {look.fade:+d}",
        f"  Saettigung:      {look.saturation:+d}",
        f"  Schaerfung:      {look.sharpness:+d}",
        f"  Schaerfungsber.: {look.sharpness_range:+d}",
        f"  Klarheit:        {look.clarity:+d}",
    ]
    if look.is_monochrome:
        lines.append(f"  (Schwarzweiss-Look)")
    return "\n".join(lines)


# Aliases für konsistente Import-Namen
canon_to_xmp = canon_to_lightroom_xmp
sony_to_xmp = sony_to_lightroom_xmp
