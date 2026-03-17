"""
DCP XML Export/Import und Profil-Operationen

Konvertiert DCP-Profile zu/von lesbarem XML-Format.
Implementiert zusätzlich die dcpTool-Operationen:
- Make Invariant: Mergt LookTable in HueSatMap
- UnTwist: Entfernt helligkeitsabhängige Farbverschiebungen

Inspiriert von dcpTool (Sandy McGuffog, GPL v2), komplett in Python umgesetzt.
"""

import struct
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional
import numpy as np

from dcp_io import (
    DCPProfile, DCPReader, DCPWriter,
    ILLUMINANT_NAMES,
    _float_to_srational, _srational_to_float,
)


# ── HueSatMap Hilfsfunktionen ─────────────────────────────────

def _parse_hue_sat_map(data: bytes, dims: tuple, endian: str = '<') -> np.ndarray:
    """
    Parst HueSatMap-Rohdaten in ein 4D numpy-Array.

    Returns:
        Array mit Shape (hueDivs, satDivs, valDivs, 3)
        wobei Kanal 0=HueShift, 1=SatScale, 2=ValScale
    """
    hue_divs, sat_divs, val_divs = dims
    num_entries = hue_divs * sat_divs * val_divs
    floats = struct.unpack(f'{endian}{num_entries * 3}f', data[:num_entries * 12])
    return np.array(floats, dtype=np.float32).reshape(hue_divs, sat_divs, val_divs, 3)


def _pack_hue_sat_map(arr: np.ndarray, endian: str = '<') -> bytes:
    """Packt ein HueSatMap-Array zurück in Bytes."""
    flat = arr.flatten()
    return struct.pack(f'{endian}{len(flat)}f', *flat)


def _interpolate_hue_sat(arr: np.ndarray, h_frac: float, s_frac: float,
                          v_frac: float) -> np.ndarray:
    """
    Trilineare Interpolation in einer HueSatMap.

    Args:
        arr: Shape (hueDivs, satDivs, valDivs, 3)
        h_frac, s_frac, v_frac: Fließkomma-Indizes (0..hueDivs etc.)

    Returns:
        Interpoliertes (3,) Array [hueShift, satScale, valScale]
    """
    hd, sd, vd = arr.shape[:3]

    h0 = int(h_frac) % hd
    h1 = (h0 + 1) % hd  # Hue wraps around
    hf = h_frac - int(h_frac)

    s0 = min(int(s_frac), sd - 1)
    s1 = min(s0 + 1, sd - 1)
    sf = s_frac - s0 if s0 < sd - 1 else 0.0

    v0 = min(int(v_frac), vd - 1)
    v1 = min(v0 + 1, vd - 1)
    vf = v_frac - v0 if v0 < vd - 1 else 0.0

    # Trilineare Interpolation
    result = np.zeros(3, dtype=np.float32)
    for hi, hw in [(h0, 1.0 - hf), (h1, hf)]:
        for si, sw in [(s0, 1.0 - sf), (s1, sf)]:
            for vi, vw in [(v0, 1.0 - vf), (v1, vf)]:
                result += arr[hi, si, vi] * (hw * sw * vw)
    return result


# ── Make Invariant ────────────────────────────────────────────

def make_invariant(profile: DCPProfile) -> DCPProfile:
    """
    Mergt die LookTable in die HueSatMap-Tabellen.

    Das macht das Profil unabhängig vom LookTable-Verarbeitungsschritt
    und damit vorhersagbarer über verschiedene RAW-Prozessoren.

    Modifiziert das Profil in-place und gibt es zurück.
    """
    if profile.look_table_dims is None or profile.look_table_data is None:
        return profile  # Nichts zu tun

    look_arr = _parse_hue_sat_map(profile.look_table_data, profile.look_table_dims)
    l_hd, l_sd, l_vd = profile.look_table_dims

    def _combine(hue_sat_data, hue_sat_dims):
        if hue_sat_data is None or hue_sat_dims is None:
            # Kein HueSatMap → LookTable wird zum HueSatMap
            return profile.look_table_data, profile.look_table_dims

        src_arr = _parse_hue_sat_map(hue_sat_data, hue_sat_dims)
        s_hd, s_sd, s_vd = hue_sat_dims

        # Gemeinsame (maximale) Auflösung
        m_hd = max(s_hd, l_hd)
        m_sd = max(s_sd, l_sd)
        m_vd = max(s_vd, l_vd)

        result = np.zeros((m_hd, m_sd, m_vd, 3), dtype=np.float32)

        for mh in range(m_hd):
            for ms in range(m_sd):
                for mv in range(m_vd):
                    # Normalisierte Koordinaten
                    h_norm = mh / m_hd * 360.0
                    s_norm = ms / (m_sd - 1) if m_sd > 1 else 0.0
                    v_norm = mv / (m_vd - 1) if m_vd > 1 else 0.0

                    # In Quell-HueSatMap interpolieren
                    h_src = h_norm / 360.0 * s_hd
                    s_src = s_norm * (s_sd - 1) if s_sd > 1 else 0.0
                    v_src = v_norm * (s_vd - 1) if s_vd > 1 else 0.0
                    hsm_val = _interpolate_hue_sat(src_arr, h_src, s_src, v_src)

                    # In LookTable interpolieren
                    h_look = h_norm / 360.0 * l_hd
                    s_look = s_norm * (l_sd - 1) if l_sd > 1 else 0.0
                    v_look = v_norm * (l_vd - 1) if l_vd > 1 else 0.0
                    look_val = _interpolate_hue_sat(look_arr, h_look, s_look, v_look)

                    # Kombinieren: Hue addiert, Sat/Val multipliziert
                    dh = hsm_val[0] + look_val[0]
                    if dh > 180.0:
                        dh -= 360.0
                    if dh < -180.0:
                        dh += 360.0
                    ds = hsm_val[1] * look_val[1]
                    dv = hsm_val[2] * look_val[2]

                    # Sat=0 muss ValScale=1.0 haben
                    if ms == 0:
                        dv = 1.0

                    result[mh, ms, mv] = [dh, ds, dv]

        new_dims = (m_hd, m_sd, m_vd)
        new_data = _pack_hue_sat_map(result)
        return new_data, new_dims

    # HueSatDeltas1
    new_data1, new_dims = _combine(
        profile.hue_sat_map_data_1, profile.hue_sat_map_dims)
    profile.hue_sat_map_data_1 = new_data1
    profile.hue_sat_map_dims = new_dims

    # HueSatDeltas2
    if profile.hue_sat_map_data_2 is not None:
        new_data2, _ = _combine(
            profile.hue_sat_map_data_2, profile.hue_sat_map_dims)
        profile.hue_sat_map_data_2 = new_data2

    # LookTable entfernen
    profile.look_table_dims = None
    profile.look_table_data = None

    if not profile.profile_name.endswith(" Invariant"):
        profile.profile_name += " Invariant"

    return profile


# ── UnTwist ──────────────────────────────────────────────────

# CC2 Patch HSV Value für Hauttonreferenz (aus dcpTool)
_TWIST_VALUE_SELECT = 0.4257


def _untwist_table(data: bytes, dims: tuple) -> tuple:
    """Kollabiert die Value-Dimension einer HueSatMap."""
    hd, sd, vd = dims
    if vd <= 1:
        return data, dims  # Schon eindimenisonal

    arr = _parse_hue_sat_map(data, dims)

    # Interpolationsposition in der Value-Dimension
    v_pos = _TWIST_VALUE_SELECT * (vd - 1)
    v_floor = min(int(v_pos), vd - 1)
    v_ceil = min(v_floor + 1, vd - 1)

    if v_floor != v_ceil:
        floor_val = v_floor / (vd - 1)
        ceil_val = v_ceil / (vd - 1)
        floor_weight = (ceil_val - _TWIST_VALUE_SELECT) / (ceil_val - floor_val)
        floor_weight = max(0.0, min(1.0, floor_weight))
    else:
        floor_weight = 1.0

    # Neues Array mit valDivs=1
    result = np.zeros((hd, sd, 1, 3), dtype=np.float32)

    for h in range(hd):
        for s in range(sd):
            floor_item = arr[h, s, v_floor]
            ceil_item = arr[h, s, v_ceil]

            interpolated = floor_item * floor_weight + ceil_item * (1.0 - floor_weight)

            # Sat=0 muss ValScale=1.0 haben
            if s == 0:
                interpolated[2] = 1.0

            result[h, s, 0] = interpolated

    new_dims = (hd, sd, 1)
    new_data = _pack_hue_sat_map(result)
    return new_data, new_dims


def untwist(profile: DCPProfile) -> DCPProfile:
    """
    Entfernt helligkeitsabhängige Farbverschiebungen (Hue Twists).

    Kollabiert die Value-Dimension der HueSatMap- und LookTable-Tabellen
    auf einen einzelnen Wert, interpoliert bei der CC2-Patch-Helligkeit.

    Modifiziert das Profil in-place und gibt es zurück.
    """
    changed = False

    if (profile.look_table_dims is not None and profile.look_table_data is not None
            and profile.look_table_dims[2] > 1):
        profile.look_table_data, profile.look_table_dims = _untwist_table(
            profile.look_table_data, profile.look_table_dims)
        changed = True

    if (profile.hue_sat_map_dims is not None and profile.hue_sat_map_data_1 is not None
            and profile.hue_sat_map_dims[2] > 1):
        old_dims = profile.hue_sat_map_dims
        profile.hue_sat_map_data_1, profile.hue_sat_map_dims = _untwist_table(
            profile.hue_sat_map_data_1, old_dims)

        if profile.hue_sat_map_data_2 is not None:
            profile.hue_sat_map_data_2, _ = _untwist_table(
                profile.hue_sat_map_data_2, old_dims)
        changed = True

    if changed and not profile.profile_name.endswith(" Untwist"):
        profile.profile_name += " Untwist"

    return profile


# ── DCP → XML Export ─────────────────────────────────────────

def _matrix_to_xml(parent: ET.Element, matrix: np.ndarray, name: str):
    """Serialisiert eine Matrix als XML-Element."""
    if matrix is None:
        return
    el = ET.SubElement(parent, name)
    rows, cols = matrix.shape
    el.set("Rows", str(rows))
    el.set("Cols", str(cols))
    for r in range(rows):
        for c in range(cols):
            item = ET.SubElement(el, "Element")
            item.set("Row", str(r))
            item.set("Col", str(c))
            item.text = f"{matrix[r, c]:.6f}"


def _hue_sat_map_to_xml(parent: ET.Element, data: bytes, dims: tuple, name: str):
    """Serialisiert eine HueSatMap als XML-Element."""
    if data is None or dims is None:
        return
    hd, sd, vd = dims
    arr = _parse_hue_sat_map(data, dims)
    el = ET.SubElement(parent, name)
    el.set("hueDivisions", str(hd))
    el.set("satDivisions", str(sd))
    el.set("valDivisions", str(vd))

    for h in range(hd):
        for s in range(sd):
            for v in range(vd):
                item = ET.SubElement(el, "Element")
                item.set("HueDiv", str(h))
                item.set("SatDiv", str(s))
                item.set("ValDiv", str(v))
                item.set("HueShift", f"{arr[h, s, v, 0]:.6f}")
                item.set("SatScale", f"{arr[h, s, v, 1]:.6f}")
                item.set("ValScale", f"{arr[h, s, v, 2]:.6f}")


def _tone_curve_to_xml(parent: ET.Element, data: bytes, count: int):
    """Serialisiert eine Tonkurve als XML-Element."""
    if data is None or count == 0:
        return
    num_points = count // 2
    floats = struct.unpack(f'<{count}f', data[:count * 4])
    el = ET.SubElement(parent, "ToneCurve")
    el.set("Size", str(num_points))
    for i in range(num_points):
        item = ET.SubElement(el, "Element")
        item.set("N", str(i))
        item.set("h", f"{floats[i * 2]:.6f}")
        item.set("v", f"{floats[i * 2 + 1]:.6f}")


def dcp_to_xml(profile: DCPProfile) -> str:
    """
    Konvertiert ein DCPProfile in XML-String.

    Das XML-Format ist kompatibel mit dcpTool.
    """
    root = ET.Element("dcpData")

    # Strings
    if profile.profile_name:
        ET.SubElement(root, "ProfileName").text = profile.profile_name
    ET.SubElement(root, "CalibrationIlluminant1").text = str(profile.illuminant_1)
    if profile.has_dual_illuminant():
        ET.SubElement(root, "CalibrationIlluminant2").text = str(profile.illuminant_2)

    # Matrizen
    _matrix_to_xml(root, profile.color_matrix_1, "ColorMatrix1")
    _matrix_to_xml(root, profile.color_matrix_2, "ColorMatrix2")
    _matrix_to_xml(root, profile.forward_matrix_1, "ForwardMatrix1")
    _matrix_to_xml(root, profile.forward_matrix_2, "ForwardMatrix2")
    _matrix_to_xml(root, profile.reduction_matrix_1, "ReductionMatrix1")
    _matrix_to_xml(root, profile.reduction_matrix_2, "ReductionMatrix2")

    # Strings und Werte
    if profile.copyright:
        ET.SubElement(root, "Copyright").text = profile.copyright
    ET.SubElement(root, "EmbedPolicy").text = str(profile.embed_policy)

    # HueSatMap Encoding
    if profile.hue_sat_map_encoding != 0:
        ET.SubElement(root, "ProfileHueSatMapEncoding").text = str(
            profile.hue_sat_map_encoding)

    # HueSatMaps
    _hue_sat_map_to_xml(root, profile.hue_sat_map_data_1,
                         profile.hue_sat_map_dims, "HueSatDeltas1")
    _hue_sat_map_to_xml(root, profile.hue_sat_map_data_2,
                         profile.hue_sat_map_dims, "HueSatDeltas2")

    # LookTable
    _hue_sat_map_to_xml(root, profile.look_table_data,
                         profile.look_table_dims, "LookTable")

    # ToneCurve
    _tone_curve_to_xml(root, profile.tone_curve_data, profile.tone_curve_count)

    # Zusätzliche Strings
    if profile.calibration_signature:
        ET.SubElement(root, "ProfileCalibrationSignature").text = (
            profile.calibration_signature)
    if profile.camera_model:
        ET.SubElement(root, "UniqueCameraModelRestriction").text = (
            profile.camera_model)

    # LookTable Encoding
    if profile.look_table_encoding != 0:
        ET.SubElement(root, "ProfileLookTableEncoding").text = str(
            profile.look_table_encoding)

    # BaselineExposureOffset
    if profile.baseline_exposure_offset != 0.0:
        ET.SubElement(root, "BaselineExposureOffset").text = f"{profile.baseline_exposure_offset:.6f}"

    # DefaultBlackRender
    if profile.default_black_render != 0:
        ET.SubElement(root, "DefaultBlackRender").text = str(
            profile.default_black_render)

    # Pretty-print
    rough = ET.tostring(root, encoding='unicode')
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding=None)


def export_dcp_to_xml(dcp_path: str, xml_path: str):
    """Exportiert eine DCP-Datei als XML."""
    profile = DCPReader().read(dcp_path)
    xml_str = dcp_to_xml(profile)
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    return profile


# ── XML → DCP Import ────────────────────────────────────────

def _xml_to_matrix(el: ET.Element) -> np.ndarray:
    """Parst ein Matrix-XML-Element."""
    rows = int(el.get("Rows", "3"))
    cols = int(el.get("Cols", "3"))
    mat = np.zeros((rows, cols), dtype=np.float64)
    for item in el.findall("Element"):
        r = int(item.get("Row", "0"))
        c = int(item.get("Col", "0"))
        mat[r, c] = float(item.text.strip())
    return mat


def _xml_to_hue_sat_map(el: ET.Element) -> tuple:
    """Parst ein HueSatMap-XML-Element. Gibt (data_bytes, dims) zurück."""
    hd = int(el.get("hueDivisions", "0"))
    sd = int(el.get("satDivisions", "0"))
    vd = int(el.get("valDivisions", "0"))
    if hd == 0 or sd == 0:
        return None, None

    arr = np.zeros((hd, sd, max(vd, 1), 3), dtype=np.float32)
    for item in el.findall("Element"):
        h = int(item.get("HueDiv", "0"))
        s = int(item.get("SatDiv", "0"))
        v = int(item.get("ValDiv", "0"))
        arr[h, s, v, 0] = float(item.get("HueShift", "0"))
        arr[h, s, v, 1] = float(item.get("SatScale", "1"))
        arr[h, s, v, 2] = float(item.get("ValScale", "1"))

    dims = (hd, sd, max(vd, 1))
    data = _pack_hue_sat_map(arr)
    return data, dims


def _xml_to_tone_curve(el: ET.Element) -> tuple:
    """Parst ein ToneCurve-XML-Element. Gibt (data_bytes, count) zurück."""
    size = int(el.get("Size", "0"))
    if size == 0:
        return None, 0

    points = []
    for item in el.findall("Element"):
        h = float(item.get("h", "0"))
        v = float(item.get("v", "0"))
        points.append((h, v))

    # Nach N sortieren
    points.sort(key=lambda p: p[0])

    count = len(points) * 2
    floats = []
    for h, v in points:
        floats.extend([h, v])
    data = struct.pack(f'<{count}f', *floats)
    return data, count


def xml_to_dcp(xml_str: str) -> DCPProfile:
    """
    Konvertiert einen XML-String zu einem DCPProfile.

    Das XML-Format ist kompatibel mit dcpTool.
    """
    root = ET.fromstring(xml_str)
    profile = DCPProfile()

    for child in root:
        tag = child.tag
        text = (child.text or "").strip()

        if tag == "ProfileName":
            profile.profile_name = text
        elif tag == "CalibrationIlluminant1":
            profile.illuminant_1 = int(text)
        elif tag == "CalibrationIlluminant2":
            profile.illuminant_2 = int(text)
        elif tag == "ColorMatrix1":
            profile.color_matrix_1 = _xml_to_matrix(child)
        elif tag == "ColorMatrix2":
            profile.color_matrix_2 = _xml_to_matrix(child)
        elif tag == "ForwardMatrix1":
            profile.forward_matrix_1 = _xml_to_matrix(child)
        elif tag == "ForwardMatrix2":
            profile.forward_matrix_2 = _xml_to_matrix(child)
        elif tag == "ReductionMatrix1":
            profile.reduction_matrix_1 = _xml_to_matrix(child)
        elif tag == "ReductionMatrix2":
            profile.reduction_matrix_2 = _xml_to_matrix(child)
        elif tag == "Copyright":
            profile.copyright = text
        elif tag == "EmbedPolicy":
            profile.embed_policy = int(text)
        elif tag == "ProfileHueSatMapEncoding":
            profile.hue_sat_map_encoding = int(text)
        elif tag == "HueSatDeltas1":
            data, dims = _xml_to_hue_sat_map(child)
            profile.hue_sat_map_data_1 = data
            profile.hue_sat_map_dims = dims
        elif tag == "HueSatDeltas2":
            data, dims = _xml_to_hue_sat_map(child)
            profile.hue_sat_map_data_2 = data
            # dims sollte gleich sein wie bei Deltas1
            if profile.hue_sat_map_dims is None:
                profile.hue_sat_map_dims = dims
        elif tag == "LookTable":
            data, dims = _xml_to_hue_sat_map(child)
            profile.look_table_data = data
            profile.look_table_dims = dims
        elif tag == "ToneCurve":
            data, count = _xml_to_tone_curve(child)
            profile.tone_curve_data = data
            profile.tone_curve_count = count
        elif tag == "ProfileCalibrationSignature":
            profile.calibration_signature = text
        elif tag == "UniqueCameraModelRestriction":
            profile.camera_model = text
        elif tag == "ProfileLookTableEncoding":
            profile.look_table_encoding = int(text)
        elif tag == "BaselineExposureOffset":
            profile.baseline_exposure_offset = float(text)
        elif tag == "DefaultBlackRender":
            profile.default_black_render = int(text)

    return profile


def import_xml_to_dcp(xml_path: str, dcp_path: str) -> DCPProfile:
    """Importiert eine XML-Datei und schreibt sie als DCP."""
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    profile = xml_to_dcp(xml_str)
    DCPWriter().write(dcp_path, profile)
    return profile


# ── CLI ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Verwendung: python dcp_xml.py <operation> <eingabe> [ausgabe]")
        print()
        print("Operationen:")
        print("  -d  DCP → XML decompilieren (Standard)")
        print("  -c  XML → DCP compilieren")
        print("  -i  DCP → DCP invariant machen (LookTable mergen)")
        print("  -u  DCP → DCP untwisten (Value-Dimension entfernen)")
        sys.exit(1)

    op = '-d'
    files = []
    for arg in sys.argv[1:]:
        if arg in ('-d', '-c', '-i', '-u'):
            op = arg
        else:
            files.append(arg)

    if op == '-d':
        profile = export_dcp_to_xml(files[0], files[1] if len(files) > 1 else '-')
        if len(files) <= 1:
            print(dcp_to_xml(profile))
        else:
            print(f"XML exportiert: {files[1]}")

    elif op == '-c':
        if len(files) < 2:
            print("Fehler: XML- und DCP-Pfad benötigt")
            sys.exit(1)
        profile = import_xml_to_dcp(files[0], files[1])
        print(f"DCP kompiliert: {files[1]}")

    elif op == '-i':
        if len(files) < 2:
            print("Fehler: Quell- und Ziel-DCP-Pfad benötigt")
            sys.exit(1)
        profile = DCPReader().read(files[0])
        make_invariant(profile)
        DCPWriter().write(files[1], profile)
        print(f"Invariant DCP: {files[1]}")

    elif op == '-u':
        if len(files) < 2:
            print("Fehler: Quell- und Ziel-DCP-Pfad benötigt")
            sys.exit(1)
        profile = DCPReader().read(files[0])
        untwist(profile)
        DCPWriter().write(files[1], profile)
        print(f"Untwisted DCP: {files[1]}")
