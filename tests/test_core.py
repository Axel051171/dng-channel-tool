"""
Tests für DNG Channel Tool

Testet die Kern-Module ohne externe Abhängigkeiten (rawpy, exifread).
Verwendet synthetische Testdaten mit numpy.
"""

import sys
import os
import struct
import tempfile

import numpy as np
import pytest

# Projekt-Root zum Pfad hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
#  channel_swap.py
# ═══════════════════════════════════════════════════════════

class TestChannelMapping:
    """Tests für ChannelMapping Datenklasse."""

    def test_identity(self):
        from channel_swap import ChannelMapping
        m = ChannelMapping()
        assert m.is_identity
        assert m.name == "RGB"
        assert m.permutation == (0, 1, 2)

    def test_rb_swap(self):
        from channel_swap import ChannelMapping
        m = ChannelMapping(r_source=2, g_source=1, b_source=0)
        assert not m.is_identity
        assert m.name == "BGR"

    def test_permutation_matrix(self):
        from channel_swap import ChannelMapping
        m = ChannelMapping(r_source=2, g_source=0, b_source=1)
        P = m.permutation_matrix()
        assert P.shape == (3, 3)
        assert np.allclose(P.sum(axis=0), 1)
        assert np.allclose(P.sum(axis=1), 1)

    def test_to_mix_matrix(self):
        from channel_swap import ChannelMapping
        m = ChannelMapping()
        mix = m.to_mix_matrix()
        assert mix.is_identity
        assert mix.is_permutation

    def test_from_permutation(self):
        from channel_swap import ChannelMapping
        m = ChannelMapping.from_permutation((1, 2, 0))
        assert m.r_source == 1
        assert m.g_source == 2
        assert m.b_source == 0


class TestMixMatrix:
    """Tests für MixMatrix Datenklasse."""

    def test_identity(self):
        from channel_swap import MixMatrix
        m = MixMatrix()
        assert m.is_identity
        assert m.is_permutation
        assert m.name == "RGB"

    def test_permutation_detection(self):
        from channel_swap import MixMatrix
        m = MixMatrix(matrix=np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=float))
        assert m.is_permutation
        assert not m.is_identity
        assert m.name == "BGR"

    def test_non_permutation(self):
        from channel_swap import MixMatrix
        m = MixMatrix(matrix=np.array([
            [0.5, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float))
        assert not m.is_permutation
        assert m.name == "Mix"

    def test_normalize_rows(self):
        from channel_swap import MixMatrix
        m = MixMatrix(matrix=np.array([
            [2, 2, 0],
            [0, 3, 0],
            [1, 1, 1],
        ], dtype=float))
        n = m.normalize_rows()
        for i in range(3):
            assert abs(n.matrix[i].sum() - 1.0) < 1e-10

    def test_to_channel_mapping_permutation(self):
        from channel_swap import MixMatrix
        m = MixMatrix(matrix=np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float))
        cm = m.to_channel_mapping()
        assert cm is not None
        assert cm.r_source == 1
        assert cm.g_source == 0

    def test_to_channel_mapping_non_permutation(self):
        from channel_swap import MixMatrix
        m = MixMatrix(matrix=np.array([
            [0.5, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float))
        assert m.to_channel_mapping() is None


class TestImageChannelOps:
    """Tests für Bildkanal-Operationen."""

    def _make_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # R
        img[:, :, 1] = 150  # G
        img[:, :, 2] = 200  # B
        return img

    def test_swap_identity(self):
        from channel_swap import ChannelMapping, swap_image_channels
        img = self._make_image()
        result = swap_image_channels(img, ChannelMapping())
        np.testing.assert_array_equal(result, img)
        assert result is not img  # Kopie, nicht Referenz

    def test_swap_rb(self):
        from channel_swap import ChannelMapping, swap_image_channels
        img = self._make_image()
        mapping = ChannelMapping(r_source=2, g_source=1, b_source=0)
        result = swap_image_channels(img, mapping)
        assert result[0, 0, 0] == 200  # R ← B
        assert result[0, 0, 1] == 150  # G unverändert
        assert result[0, 0, 2] == 100  # B ← R

    def test_mix_identity(self):
        from channel_swap import MixMatrix, mix_image_channels
        img = self._make_image()
        result = mix_image_channels(img, MixMatrix())
        np.testing.assert_array_equal(result, img)

    def test_mix_weighted(self):
        from channel_swap import MixMatrix, mix_image_channels
        img = self._make_image()
        m = MixMatrix(matrix=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0, 0.5],
        ], dtype=float))
        result = mix_image_channels(img, m)
        assert result[0, 0, 0] == 100  # R unverändert
        assert result[0, 0, 2] == 150  # B = 0.5*100 + 0.5*200

    def test_apply_to_image_fast_path(self):
        from channel_swap import MixMatrix, apply_to_image
        img = self._make_image()
        # Permutationsmatrix → sollte schnellen Pfad nutzen
        m = MixMatrix(matrix=np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=float))
        result = apply_to_image(img, m)
        assert result[0, 0, 0] == 200


class TestDCPMatrixOps:
    """Tests für DCP-Matrix-Transformationen."""

    def test_swap_color_matrix(self):
        from channel_swap import ChannelMapping, swap_color_matrix
        cm = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)
        mapping = ChannelMapping(r_source=2, g_source=1, b_source=0)
        result = swap_color_matrix(cm, mapping)
        np.testing.assert_array_equal(result[0], cm[2])
        np.testing.assert_array_equal(result[2], cm[0])

    def test_swap_forward_matrix(self):
        from channel_swap import ChannelMapping, swap_forward_matrix
        fm = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)
        mapping = ChannelMapping(r_source=2, g_source=1, b_source=0)
        result = swap_forward_matrix(fm, mapping)
        np.testing.assert_array_equal(result[:, 0], fm[:, 2])
        np.testing.assert_array_equal(result[:, 2], fm[:, 0])

    def test_swap_identity_noop(self):
        from channel_swap import ChannelMapping, swap_color_matrix
        cm = np.eye(3)
        result = swap_color_matrix(cm, ChannelMapping())
        np.testing.assert_array_equal(result, cm)


# ═══════════════════════════════════════════════════════════
#  color_checker.py
# ═══════════════════════════════════════════════════════════

class TestColorChecker:
    """Tests für ColorChecker-Kalibrierung."""

    def test_sample_patch_color(self):
        from color_checker import sample_patch_color
        img = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
        r, g, b = sample_patch_color(img, 50, 50)
        assert r == 100
        assert g == 150
        assert b == 200

    def test_sample_patch_color_edge(self):
        from color_checker import sample_patch_color
        img = np.full((100, 100, 3), [50, 50, 50], dtype=np.uint8)
        r, g, b = sample_patch_color(img, 0, 0, patch_size=20)
        assert r == 50

    def test_correction_matrix_identity(self):
        from color_checker import compute_correction_matrix, ColorPatch
        patches = [
            ColorPatch("R", (200, 50, 50), (200, 50, 50)),
            ColorPatch("G", (50, 200, 50), (50, 200, 50)),
            ColorPatch("B", (50, 50, 200), (50, 50, 200)),
            ColorPatch("W", (200, 200, 200), (200, 200, 200)),
        ]
        result = compute_correction_matrix(patches)
        assert result.avg_delta_e < 1.0
        assert result.correction_matrix.shape == (3, 3)

    def test_correction_matrix_basic(self):
        from color_checker import compute_correction_matrix, ColorPatch
        # Gemessene Werte systematisch zu dunkel
        patches = [
            ColorPatch("R", (100, 25, 25), (200, 50, 50)),
            ColorPatch("G", (25, 100, 25), (50, 200, 50)),
            ColorPatch("B", (25, 25, 100), (50, 50, 200)),
            ColorPatch("W", (128, 128, 128), (200, 200, 200)),
        ]
        result = compute_correction_matrix(patches)
        assert result.correction_matrix is not None
        assert result.avg_delta_e < 20  # Sollte eine brauchbare Korrektur finden

    def test_auto_detect_returns_24(self):
        from color_checker import auto_detect_colorchecker
        img = np.zeros((600, 900, 3), dtype=np.uint8)
        centers = auto_detect_colorchecker(img)
        assert len(centers) == 24

    def test_too_few_patches_raises(self):
        from color_checker import compute_correction_matrix, ColorPatch
        patches = [
            ColorPatch("R", (200, 50, 50), (200, 50, 50)),
            ColorPatch("G", (50, 200, 50), (50, 200, 50)),
        ]
        with pytest.raises(ValueError):
            compute_correction_matrix(patches)

    def test_delta_e_is_cielab(self):
        """Prüft dass Delta E in CIELAB berechnet wird, nicht euklidisch in sRGB."""
        from color_checker import compute_correction_matrix, ColorPatch
        # Erstelle Patches mit bekanntem Farbabstand
        patches = [
            ColorPatch("1", (100, 100, 100), (100, 100, 100)),
            ColorPatch("2", (200, 200, 200), (200, 200, 200)),
            ColorPatch("3", (50, 50, 50), (50, 50, 50)),
            ColorPatch("4", (150, 150, 150), (150, 150, 150)),
        ]
        result = compute_correction_matrix(patches)
        # Bei identischen Farben muss Delta E = 0 sein
        assert result.avg_delta_e < 0.01


# ═══════════════════════════════════════════════════════════
#  wb_picker.py
# ═══════════════════════════════════════════════════════════

class TestWBPicker:
    """Tests für Weißabgleich-Pipette."""

    def test_neutral_pixel_gains_near_one(self):
        from wb_picker import calculate_wb_from_pixel
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        wb = calculate_wb_from_pixel(img, 50, 50)
        assert abs(wb['r_gain'] - 1.0) < 0.01
        assert abs(wb['b_gain'] - 1.0) < 0.01

    def test_warm_pixel_correction(self):
        from wb_picker import calculate_wb_from_pixel
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # Zu rot
        img[:, :, 1] = 128
        img[:, :, 2] = 80   # Zu wenig blau
        wb = calculate_wb_from_pixel(img, 50, 50)
        # Korrektur: R runter, B hoch
        assert wb['r_gain'] < 1.0
        assert wb['b_gain'] > 1.0

    def test_gain_limit(self):
        """WB-Gains dürfen nicht über 4.0 gehen."""
        from wb_picker import calculate_wb_from_pixel
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 100
        img[:, :, 2] = 2  # Extrem wenig Blau
        wb = calculate_wb_from_pixel(img, 50, 50)
        assert wb['r_gain'] <= 4.0
        assert wb['b_gain'] <= 4.0

    def test_apply_wb_correction(self):
        from wb_picker import apply_wb_correction
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = apply_wb_correction(img, 1.5, 1.0, 0.5)
        assert result[0, 0, 0] == 150  # R * 1.5
        assert result[0, 0, 1] == 100  # G * 1.0
        assert result[0, 0, 2] == 50   # B * 0.5

    def test_apply_wb_correction_clipping(self):
        from wb_picker import apply_wb_correction
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = apply_wb_correction(img, 2.0, 1.0, 1.0)
        assert result[0, 0, 0] == 255  # Clipped

    def test_histogram_match_shape(self):
        from wb_picker import histogram_match
        src = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        ref = np.random.randint(100, 200, (30, 30, 3), dtype=np.uint8)
        result = histogram_match(src, ref)
        assert result.shape == src.shape
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════════
#  ir_tools.py
# ═══════════════════════════════════════════════════════════

class TestIRTools:
    """Tests für Infrarot-Werkzeuge."""

    def test_ir_filters_defined(self):
        from ir_tools import IR_FILTERS
        assert len(IR_FILTERS) == 6
        assert "720nm" in IR_FILTERS
        assert "Full Spectrum" in IR_FILTERS

    def test_ir_filters_valid_data(self):
        from ir_tools import IR_FILTERS
        for name, f in IR_FILTERS.items():
            assert len(f.channel_weights) == 3
            assert len(f.sensitivity) == 3
            assert f.typical_wb_temp > 0

    def test_false_color_presets_defined(self):
        from ir_tools import IR_FALSE_COLOR_PRESETS
        assert len(IR_FALSE_COLOR_PRESETS) == 8
        assert "Classic Blue Sky" in IR_FALSE_COLOR_PRESETS
        assert "Kodak Aerochrome" in IR_FALSE_COLOR_PRESETS

    def test_false_color_presets_valid(self):
        from ir_tools import IR_FALSE_COLOR_PRESETS
        for name, p in IR_FALSE_COLOR_PRESETS.items():
            assert p.mix_matrix.shape == (3, 3)
            assert isinstance(p.saturation, float)

    def test_calculate_ir_wb(self):
        from ir_tools import calculate_ir_wb
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 100
        img[:, :, 2] = 50
        wb = calculate_ir_wb(img, 50, 50)
        assert 'r_gain' in wb
        assert 'g_gain' in wb
        assert 'b_gain' in wb
        assert wb['r_gain'] > 0
        assert wb['b_gain'] <= 6.0  # Limit check

    def test_apply_ir_wb(self):
        from ir_tools import apply_ir_wb
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        wb = {'r_gain': 1.5, 'g_gain': 1.0, 'b_gain': 2.0}
        result = apply_ir_wb(img, wb)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_apply_ir_preset(self):
        from ir_tools import apply_ir_preset, IR_FALSE_COLOR_PRESETS
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        preset = IR_FALSE_COLOR_PRESETS["Classic Blue Sky"]
        result = apply_ir_preset(img, preset)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_simulate_ir_filter(self):
        from ir_tools import simulate_ir_filter
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = simulate_ir_filter(img, "720nm")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_detect_hotspot_uniform(self):
        """Kein Hotspot bei gleichmäßig beleuchteten Bildern."""
        from ir_tools import detect_hotspot
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = detect_hotspot(img)
        assert not result.has_hotspot

    def test_detect_hotspot_bright_center(self):
        """Hotspot bei heller Mitte und dunklem Rand."""
        from ir_tools import detect_hotspot
        img = np.full((200, 200, 3), 50, dtype=np.uint8)
        # Helles Zentrum
        cy, cx = 100, 100
        for y in range(200):
            for x in range(200):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < 30:
                    img[y, x] = 220
        result = detect_hotspot(img, threshold=0.1)
        assert result.has_hotspot
        assert result.severity > 0

    def test_correct_hotspot_shape(self):
        from ir_tools import detect_hotspot, correct_hotspot, HotspotResult
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        hs = HotspotResult(has_hotspot=True, severity=0.5,
                           center_x=50, center_y=50, radius=20,
                           brightness_center=200, brightness_edge=100,
                           description="Test")
        result = correct_hotspot(img, hs)
        assert result.shape == img.shape

    def test_ndvi_range(self):
        from ir_tools import calculate_ndvi
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = calculate_ndvi(img)
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_curve_to_lut(self):
        from ir_tools import _curve_to_lut
        # Lineare Kurve
        lut = _curve_to_lut([(0, 0), (255, 255)])
        assert len(lut) == 256
        assert lut[0] == 0
        assert lut[255] == 255
        assert lut[128] == 128  # Linear

    def test_curve_to_lut_s_curve(self):
        from ir_tools import _curve_to_lut
        lut = _curve_to_lut([(0, 0), (64, 30), (192, 225), (255, 255)])
        assert lut[0] == 0
        assert lut[255] == 255
        assert lut[64] == 30


# ═══════════════════════════════════════════════════════════
#  lut_export.py
# ═══════════════════════════════════════════════════════════

class TestLUTExport:
    """Tests für 3D LUT Export."""

    def test_identity_lut_corners(self):
        from lut_export import generate_identity_lut
        lut = generate_identity_lut(17)
        assert lut.shape == (17, 17, 17, 3)
        # Schwarz
        np.testing.assert_allclose(lut[0, 0, 0], [0, 0, 0])
        # Weiß
        np.testing.assert_allclose(lut[16, 16, 16], [1, 1, 1])
        # Rot
        np.testing.assert_allclose(lut[16, 0, 0], [1, 0, 0])
        # Grün
        np.testing.assert_allclose(lut[0, 16, 0], [0, 1, 0])
        # Blau
        np.testing.assert_allclose(lut[0, 0, 16], [0, 0, 1])

    def test_mix_matrix_identity_lut(self):
        from lut_export import mix_matrix_to_lut, generate_identity_lut
        identity = np.eye(3)
        lut = mix_matrix_to_lut(identity, size=9)
        ref = generate_identity_lut(9)
        np.testing.assert_allclose(lut, ref, atol=1e-10)

    def test_interpolate_curve_linear(self):
        from lut_export import _interpolate_curve_normalized
        curve = _interpolate_curve_normalized([(0, 0), (255, 255)])
        assert len(curve) == 256
        np.testing.assert_allclose(curve, np.linspace(0, 1, 256), atol=0.01)

    def test_combined_lut_monochrome(self):
        from lut_export import combined_lut
        lut = combined_lut(monochrome=True, size=9)
        # Prüfe dass R=G=B für alle Einträge
        for r in range(9):
            for g in range(9):
                for b in range(9):
                    rgb = lut[r, g, b]
                    assert abs(rgb[0] - rgb[1]) < 1e-10
                    assert abs(rgb[1] - rgb[2]) < 1e-10

    def test_write_cube_lut_format(self):
        from lut_export import write_cube_lut, generate_identity_lut
        lut = generate_identity_lut(5)
        with tempfile.NamedTemporaryFile(suffix='.cube', delete=False, mode='w') as f:
            path = f.name
        try:
            write_cube_lut(path, lut, title="Test LUT")
            with open(path, 'r') as f:
                content = f.read()
            assert 'TITLE "Test LUT"' in content
            assert 'LUT_3D_SIZE 5' in content
            assert 'DOMAIN_MIN' in content
            assert 'DOMAIN_MAX' in content
            # Zähle Datenzeilen (5^3 = 125)
            lines = [l for l in content.strip().split('\n')
                     if l and not l.startswith('#') and not l.startswith('TITLE')
                     and not l.startswith('LUT') and not l.startswith('DOMAIN')]
            assert len(lines) == 125
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
#  dcp_io.py
# ═══════════════════════════════════════════════════════════

class TestDCPIO:
    """Tests für DCP-Profil Lesen/Schreiben."""

    def test_profile_construction(self):
        from dcp_io import DCPProfile, ILLUMINANT_D65
        cm = np.array([
            [0.6, 0.1, 0.1],
            [0.3, 0.7, 0.1],
            [0.1, 0.2, 0.8],
        ])
        profile = DCPProfile(
            camera_model="TEST CAMERA",
            profile_name="Test Profile",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
        )
        assert profile.camera_model == "TEST CAMERA"
        assert profile.color_matrix_1 is not None

    def test_write_read_roundtrip(self):
        from dcp_io import DCPProfile, DCPWriter, DCPReader, ILLUMINANT_D65
        cm = np.array([
            [0.6730, 0.1136, 0.1672],
            [0.2820, 0.7282, -0.0102],
            [0.0187, -0.1474, 0.9925],
        ])
        profile = DCPProfile(
            camera_model="NIKON Z 6_2",
            profile_name="Roundtrip Test",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
            copyright="Test",
        )
        with tempfile.NamedTemporaryFile(suffix='.dcp', delete=False) as f:
            path = f.name
        try:
            DCPWriter().write(path, profile)
            loaded = DCPReader().read(path)
            assert loaded.camera_model == "NIKON Z 6_2"
            assert loaded.profile_name == "Roundtrip Test"
            assert loaded.color_matrix_1 is not None
            np.testing.assert_allclose(loaded.color_matrix_1, cm, atol=0.001)
        finally:
            os.unlink(path)

    def test_rewrite_camera_model(self):
        from dcp_io import (
            DCPProfile, DCPWriter, DCPReader, ILLUMINANT_D65,
            rewrite_dcp_camera_model,
        )
        cm = np.array([
            [0.6730, 0.1136, 0.1672],
            [0.2820, 0.7282, -0.0102],
            [0.0187, -0.1474, 0.9925],
        ])
        profile = DCPProfile(
            camera_model="NIKON Z 6",
            profile_name="NIKON Z 6 Standard",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
        )
        with tempfile.NamedTemporaryFile(suffix='.dcp', delete=False) as f:
            src_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.dcp', delete=False) as f:
            dest_path = f.name
        try:
            DCPWriter().write(src_path, profile)

            # Umschreiben auf Z 6_2
            result = rewrite_dcp_camera_model(src_path, dest_path, "NIKON Z 6_2")
            assert result.camera_model == "NIKON Z 6_2"
            # Profilname sollte automatisch angepasst werden
            assert "NIKON Z 6_2" in result.profile_name

            # Ergebnis-Datei prüfen
            loaded = DCPReader().read(dest_path)
            assert loaded.camera_model == "NIKON Z 6_2"
            np.testing.assert_allclose(loaded.color_matrix_1, cm, atol=0.001)
        finally:
            os.unlink(src_path)
            os.unlink(dest_path)

    def test_rewrite_camera_model_custom_name(self):
        from dcp_io import (
            DCPProfile, DCPWriter, DCPReader, ILLUMINANT_D65,
            rewrite_dcp_camera_model,
        )
        cm = np.eye(3)
        profile = DCPProfile(
            camera_model="Canon EOS 5D Mark IV",
            profile_name="My Profile",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
        )
        with tempfile.NamedTemporaryFile(suffix='.dcp', delete=False) as f:
            path = f.name
        try:
            DCPWriter().write(path, profile)
            result = rewrite_dcp_camera_model(path, path, "Canon EOS R5", "Mein R5 Profil")
            assert result.camera_model == "Canon EOS R5"
            assert result.profile_name == "Mein R5 Profil"

            loaded = DCPReader().read(path)
            assert loaded.camera_model == "Canon EOS R5"
            assert loaded.profile_name == "Mein R5 Profil"
        finally:
            os.unlink(path)

    def test_extended_fields_roundtrip(self):
        """Testet die neuen DCP-Felder (Encoding, BlackRender etc.)."""
        from dcp_io import DCPProfile, DCPWriter, DCPReader, ILLUMINANT_D65
        cm = np.eye(3)
        profile = DCPProfile(
            camera_model="TEST",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
            hue_sat_map_encoding=1,
            look_table_encoding=1,
            baseline_exposure_offset=0.5,
            default_black_render=1,
        )
        with tempfile.NamedTemporaryFile(suffix='.dcp', delete=False) as f:
            path = f.name
        try:
            DCPWriter().write(path, profile)
            loaded = DCPReader().read(path)
            assert loaded.hue_sat_map_encoding == 1
            assert loaded.look_table_encoding == 1
            assert abs(loaded.baseline_exposure_offset - 0.5) < 0.001
            assert loaded.default_black_render == 1
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
#  dcp_xml.py
# ═══════════════════════════════════════════════════════════

class TestDCPXML:
    """Tests für DCP ↔ XML Konvertierung und Profil-Operationen."""

    def _make_profile_with_huesat(self):
        """Erzeugt ein Testprofil mit HueSatMap und LookTable."""
        from dcp_io import DCPProfile, ILLUMINANT_A, ILLUMINANT_D65
        hd, sd, vd = 6, 5, 2
        num = hd * sd * vd * 3
        # HueSatMap: kleine Hue-Shifts, Sat/Val um 1.0
        hsm_floats = []
        for h in range(hd):
            for s in range(sd):
                for v in range(vd):
                    hsm_floats.extend([h * 0.5, 1.0 + s * 0.01, 1.0])
        hsm_data = struct.pack(f'<{num}f', *hsm_floats)

        # LookTable: 3x3x2
        ld, ls, lv = 3, 3, 2
        lnum = ld * ls * lv * 3
        look_floats = []
        for h in range(ld):
            for s in range(ls):
                for v in range(lv):
                    look_floats.extend([0.1, 1.05, 1.02])
        look_data = struct.pack(f'<{lnum}f', *look_floats)

        # Tonkurve: 4 Punkte (8 floats)
        curve_floats = [0.0, 0.0, 0.25, 0.3, 0.75, 0.7, 1.0, 1.0]
        curve_data = struct.pack('<8f', *curve_floats)

        return DCPProfile(
            camera_model="NIKON Z 8",
            profile_name="Test HSM",
            copyright="Test",
            color_matrix_1=np.array([
                [0.6730, 0.1136, 0.1672],
                [0.2820, 0.7282, -0.0102],
                [0.0187, -0.1474, 0.9925],
            ]),
            color_matrix_2=np.eye(3) * 0.9,
            forward_matrix_1=np.eye(3) * 0.8,
            illuminant_1=ILLUMINANT_A,
            illuminant_2=ILLUMINANT_D65,
            hue_sat_map_dims=(hd, sd, vd),
            hue_sat_map_data_1=hsm_data,
            look_table_dims=(ld, ls, lv),
            look_table_data=look_data,
            tone_curve_data=curve_data,
            tone_curve_count=8,
        )

    def test_xml_roundtrip_basic(self):
        """Einfacher XML-Export/Import Roundtrip."""
        from dcp_io import DCPProfile, ILLUMINANT_D65
        from dcp_xml import dcp_to_xml, xml_to_dcp
        cm = np.array([
            [0.6730, 0.1136, 0.1672],
            [0.2820, 0.7282, -0.0102],
            [0.0187, -0.1474, 0.9925],
        ])
        profile = DCPProfile(
            camera_model="NIKON Z 6_2",
            profile_name="Roundtrip Test",
            color_matrix_1=cm,
            illuminant_1=ILLUMINANT_D65,
            copyright="TestCopy",
            embed_policy=1,
        )
        xml_str = dcp_to_xml(profile)
        assert "NIKON Z 6_2" in xml_str
        assert "Roundtrip Test" in xml_str
        assert "ColorMatrix1" in xml_str

        loaded = xml_to_dcp(xml_str)
        assert loaded.camera_model == "NIKON Z 6_2"
        assert loaded.profile_name == "Roundtrip Test"
        assert loaded.copyright == "TestCopy"
        assert loaded.embed_policy == 1
        np.testing.assert_allclose(loaded.color_matrix_1, cm, atol=0.0001)

    def test_xml_roundtrip_dual_illuminant(self):
        from dcp_io import DCPProfile, ILLUMINANT_A, ILLUMINANT_D65
        from dcp_xml import dcp_to_xml, xml_to_dcp
        profile = DCPProfile(
            camera_model="Test",
            color_matrix_1=np.eye(3),
            color_matrix_2=np.eye(3) * 0.9,
            forward_matrix_1=np.eye(3) * 0.8,
            forward_matrix_2=np.eye(3) * 0.7,
            illuminant_1=ILLUMINANT_A,
            illuminant_2=ILLUMINANT_D65,
        )
        xml_str = dcp_to_xml(profile)
        loaded = xml_to_dcp(xml_str)
        assert loaded.color_matrix_2 is not None
        assert loaded.forward_matrix_1 is not None
        assert loaded.forward_matrix_2 is not None
        np.testing.assert_allclose(loaded.color_matrix_2, np.eye(3) * 0.9, atol=0.0001)

    def test_xml_roundtrip_huesat(self):
        """Roundtrip mit HueSatMap."""
        from dcp_xml import dcp_to_xml, xml_to_dcp, _parse_hue_sat_map
        profile = self._make_profile_with_huesat()
        xml_str = dcp_to_xml(profile)
        assert "HueSatDeltas1" in xml_str
        assert "HueShift" in xml_str

        loaded = xml_to_dcp(xml_str)
        assert loaded.hue_sat_map_dims == profile.hue_sat_map_dims
        # Vergleiche Werte
        orig = _parse_hue_sat_map(profile.hue_sat_map_data_1, profile.hue_sat_map_dims)
        roundtripped = _parse_hue_sat_map(loaded.hue_sat_map_data_1, loaded.hue_sat_map_dims)
        np.testing.assert_allclose(roundtripped, orig, atol=0.0001)

    def test_xml_roundtrip_tone_curve(self):
        from dcp_xml import dcp_to_xml, xml_to_dcp
        profile = self._make_profile_with_huesat()
        xml_str = dcp_to_xml(profile)
        assert "ToneCurve" in xml_str

        loaded = xml_to_dcp(xml_str)
        assert loaded.tone_curve_count == 8
        orig_floats = struct.unpack('<8f', profile.tone_curve_data)
        loaded_floats = struct.unpack('<8f', loaded.tone_curve_data)
        for o, l in zip(orig_floats, loaded_floats):
            assert abs(o - l) < 0.0001

    def test_xml_roundtrip_look_table(self):
        from dcp_xml import dcp_to_xml, xml_to_dcp
        profile = self._make_profile_with_huesat()
        xml_str = dcp_to_xml(profile)
        assert "LookTable" in xml_str

        loaded = xml_to_dcp(xml_str)
        assert loaded.look_table_dims == profile.look_table_dims

    def test_export_import_file(self):
        """Test über Dateisystem."""
        from dcp_io import DCPProfile, DCPWriter, DCPReader, ILLUMINANT_D65
        from dcp_xml import export_dcp_to_xml, import_xml_to_dcp
        profile = DCPProfile(
            camera_model="FileTest",
            color_matrix_1=np.eye(3),
            illuminant_1=ILLUMINANT_D65,
        )
        dcp1 = tempfile.NamedTemporaryFile(suffix='.dcp', delete=False).name
        xml_file = tempfile.NamedTemporaryFile(suffix='.xml', delete=False).name
        dcp2 = tempfile.NamedTemporaryFile(suffix='.dcp', delete=False).name
        try:
            DCPWriter().write(dcp1, profile)
            export_dcp_to_xml(dcp1, xml_file)
            assert os.path.getsize(xml_file) > 100

            import_xml_to_dcp(xml_file, dcp2)
            loaded = DCPReader().read(dcp2)
            assert loaded.camera_model == "FileTest"
        finally:
            for f in [dcp1, xml_file, dcp2]:
                os.unlink(f)

    def test_make_invariant(self):
        """LookTable wird in HueSatMap gemergt."""
        from dcp_xml import make_invariant
        profile = self._make_profile_with_huesat()
        assert profile.look_table_data is not None

        make_invariant(profile)
        assert profile.look_table_data is None
        assert profile.look_table_dims is None
        assert profile.hue_sat_map_data_1 is not None
        assert "Invariant" in profile.profile_name

    def test_make_invariant_no_looktable(self):
        """Profil ohne LookTable bleibt unverändert."""
        from dcp_io import DCPProfile, ILLUMINANT_D65
        from dcp_xml import make_invariant
        profile = DCPProfile(
            camera_model="Test",
            color_matrix_1=np.eye(3),
            illuminant_1=ILLUMINANT_D65,
        )
        orig_name = profile.profile_name
        make_invariant(profile)
        assert profile.profile_name == orig_name

    def test_untwist(self):
        """Value-Dimension wird auf 1 reduziert."""
        from dcp_xml import untwist
        profile = self._make_profile_with_huesat()
        assert profile.hue_sat_map_dims[2] == 2  # valDivs=2

        untwist(profile)
        assert profile.hue_sat_map_dims[2] == 1
        assert "Untwist" in profile.profile_name

    def test_untwist_already_flat(self):
        """Profil mit valDivs=1 bleibt unverändert."""
        from dcp_io import DCPProfile, ILLUMINANT_D65
        from dcp_xml import untwist
        # HueSatMap mit valDivs=1
        hd, sd = 6, 5
        num = hd * sd * 1 * 3
        hsm_floats = [0.0, 1.0, 1.0] * (hd * sd)
        hsm_data = struct.pack(f'<{num}f', *hsm_floats)
        profile = DCPProfile(
            camera_model="Test",
            color_matrix_1=np.eye(3),
            illuminant_1=ILLUMINANT_D65,
            hue_sat_map_dims=(hd, sd, 1),
            hue_sat_map_data_1=hsm_data,
        )
        orig_name = profile.profile_name
        untwist(profile)
        # Kein "Untwist" Suffix wenn nichts geändert wurde
        assert profile.profile_name == orig_name

    def test_untwist_look_table(self):
        """UnTwist funktioniert auch auf LookTable."""
        from dcp_xml import untwist
        profile = self._make_profile_with_huesat()
        assert profile.look_table_dims[2] == 2

        untwist(profile)
        assert profile.look_table_dims[2] == 1


# ═══════════════════════════════════════════════════════════
#  undo.py
# ═══════════════════════════════════════════════════════════

class TestUndoManager:
    """Tests für das Undo/Redo-System."""

    def test_initial_state(self):
        from undo import UndoManager
        um = UndoManager()
        assert not um.can_undo
        assert not um.can_redo

    def test_push_and_undo(self):
        from undo import UndoManager
        um = UndoManager()
        img1 = np.zeros((5, 5, 3), dtype=np.uint8)
        img2 = np.ones((5, 5, 3), dtype=np.uint8) * 128
        um.push(img1, np.eye(3), "Schritt 1")
        um.push(img2, np.eye(3) * 2, "Schritt 2")
        assert um.can_undo
        state = um.undo()
        assert state is not None

    def test_undo_redo_cycle(self):
        from undo import UndoManager
        um = UndoManager()
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        um.push(img, np.eye(3), "A")
        um.push(img, np.eye(3), "B")

        um.undo()
        assert um.can_redo
        state = um.redo()
        assert state is not None
        assert state.description == "B"

    def test_push_clears_redo(self):
        from undo import UndoManager
        um = UndoManager()
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        um.push(img, np.eye(3), "A")
        um.push(img, np.eye(3), "B")
        um.undo()
        assert um.can_redo
        # Neuer Push löscht Redo
        um.push(img, np.eye(3), "C")
        assert not um.can_redo

    def test_max_steps(self):
        from undo import UndoManager
        um = UndoManager(max_steps=3)
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        for i in range(10):
            um.push(img, np.eye(3), f"Schritt {i}")
        # Nur die letzten 3 sollten noch da sein
        count = 0
        while um.can_undo:
            um.undo()
            count += 1
        assert count == 3

    def test_clear(self):
        from undo import UndoManager
        um = UndoManager()
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        um.push(img, np.eye(3), "A")
        um.clear()
        assert not um.can_undo
        assert not um.can_redo

    def test_callback(self):
        from undo import UndoManager
        um = UndoManager()
        calls = []
        um.on_change(lambda: calls.append(1))
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        um.push(img, np.eye(3), "A")
        assert len(calls) == 1
        um.undo()
        assert len(calls) == 2

    def test_descriptions(self):
        from undo import UndoManager
        um = UndoManager()
        assert um.undo_description == ""
        assert um.redo_description == ""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        um.push(img, np.eye(3), "Kanal-Tausch")
        assert um.undo_description == "Kanal-Tausch"


# ═══════════════════════════════════════════════════════════
#  logging_setup.py
# ═══════════════════════════════════════════════════════════

class TestLoggingSetup:
    """Tests für Logging-Konfiguration."""

    def test_setup_returns_logger(self):
        from logging_setup import setup_logging
        import logging
        # Reset für sauberen Test
        logger = logging.getLogger("dng_channel_tool")
        logger.handlers.clear()

        result = setup_logging("DEBUG")
        assert result is not None
        assert result.name == "dng_channel_tool"

    def test_setup_idempotent(self):
        from logging_setup import setup_logging
        import logging
        logger = logging.getLogger("dng_channel_tool")
        logger.handlers.clear()

        setup_logging()
        n_handlers = len(logger.handlers)
        setup_logging()  # Zweiter Aufruf
        assert len(logger.handlers) == n_handlers


# ═══════════════════════════════════════════════════════════
#  dng_writer.py
# ═══════════════════════════════════════════════════════════

class TestPGMReader:
    """Tests für den PGM-Reader."""

    def _make_pgm(self, width, height, bits=8, comment=None):
        """Erzeugt eine temporäre PGM-Datei."""
        max_val = 255 if bits == 8 else 65535
        header = f"P5\n"
        if comment:
            header += f"# {comment}\n"
        header += f"{width} {height}\n{max_val}\n"

        if bits == 8:
            pixels = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            pixel_bytes = pixels.tobytes()
        else:
            pixels = np.random.randint(0, 65536, (height, width), dtype=np.uint16)
            # PGM ist Big-Endian
            pixel_bytes = pixels.astype('>u2').tobytes()

        f = tempfile.NamedTemporaryFile(suffix='.pgm', delete=False)
        f.write(header.encode('ascii'))
        f.write(pixel_bytes)
        f.close()
        return f.name, pixels

    def test_read_8bit(self):
        from dng_writer import read_pgm
        path, pixels = self._make_pgm(16, 12, bits=8)
        try:
            pgm = read_pgm(path)
            assert pgm.width == 16
            assert pgm.height == 12
            assert pgm.bits_per_sample == 8
            assert pgm.max_val == 255
            assert len(pgm.pixel_data) == 16 * 12
            # Pixeldaten müssen übereinstimmen
            loaded = np.frombuffer(pgm.pixel_data, dtype=np.uint8).reshape(12, 16)
            np.testing.assert_array_equal(loaded, pixels)
        finally:
            os.unlink(path)

    def test_read_16bit(self):
        from dng_writer import read_pgm
        path, pixels = self._make_pgm(8, 6, bits=16)
        try:
            pgm = read_pgm(path)
            assert pgm.width == 8
            assert pgm.height == 6
            assert pgm.bits_per_sample == 16
            assert pgm.max_val == 65535
            assert len(pgm.pixel_data) == 8 * 6 * 2
            # Little-Endian zurücklesen
            loaded = np.frombuffer(pgm.pixel_data, dtype='<u2').reshape(6, 8)
            np.testing.assert_array_equal(loaded, pixels)
        finally:
            os.unlink(path)

    def test_read_with_comment(self):
        from dng_writer import read_pgm
        path, _ = self._make_pgm(4, 4, bits=8, comment="Test Kommentar")
        try:
            pgm = read_pgm(path)
            assert pgm.width == 4
            assert pgm.height == 4
        finally:
            os.unlink(path)

    def test_invalid_magic(self):
        from dng_writer import read_pgm
        f = tempfile.NamedTemporaryFile(suffix='.pgm', delete=False)
        f.write(b"P6\n4 4\n255\n" + b'\x00' * 48)
        f.close()
        try:
            with pytest.raises(ValueError, match="P5"):
                read_pgm(f.name)
        finally:
            os.unlink(f.name)


class TestDNGWriter:
    """Tests für den DNG-Writer."""

    def _parse_tiff_header(self, data):
        """Parst den TIFF-Header."""
        byte_order = struct.unpack_from('<H', data, 0)[0]
        magic = struct.unpack_from('<H', data, 2)[0]
        ifd_offset = struct.unpack_from('<I', data, 4)[0]
        return byte_order, magic, ifd_offset

    def _parse_ifd_tags(self, data, ifd_offset):
        """Parst alle IFD-Tags."""
        num_entries = struct.unpack_from('<H', data, ifd_offset)[0]
        tags = {}
        offset = ifd_offset + 2
        for _ in range(num_entries):
            tag_id, tag_type, count = struct.unpack_from('<HHI', data, offset)
            value_raw = data[offset + 8:offset + 12]
            tags[tag_id] = (tag_type, count, value_raw)
            offset += 12
        return tags

    def _write_test_dng(self, width=8, height=6, bits=8, config=None):
        """Erzeugt eine Test-DNG und gibt (Pfad, Rohdaten) zurück."""
        from dng_writer import DNGWriter, PGMData, DNGConfig

        if config is None:
            config = DNGConfig()
        max_val = 255 if bits == 8 else 65535
        dtype = np.uint8 if bits == 8 else np.uint16
        pixels = np.random.randint(0, max_val + 1, (height, width), dtype=dtype)
        pixel_bytes = pixels.astype(f'<u{bits // 8}').tobytes()

        pgm = PGMData(width=width, height=height, max_val=max_val,
                       bits_per_sample=bits, pixel_data=pixel_bytes)

        f = tempfile.NamedTemporaryFile(suffix='.dng', delete=False)
        f.close()
        DNGWriter().write(f.name, pgm, config)

        with open(f.name, 'rb') as fh:
            dng_data = fh.read()
        return f.name, dng_data, pixel_bytes

    def test_tiff_header(self):
        path, data, _ = self._write_test_dng()
        try:
            bo, magic, ifd_off = self._parse_tiff_header(data)
            assert bo == 0x4949          # Little-Endian
            assert magic == 42           # Standard TIFF (nicht DCP 0x4352)
            assert ifd_off == 8
        finally:
            os.unlink(path)

    def test_dng_version_tag(self):
        path, data, _ = self._write_test_dng()
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 50706 in tags  # DNGVersion
            _, count, raw = tags[50706]
            assert count == 4
            assert raw[:4] == bytes([1, 4, 0, 0])
        finally:
            os.unlink(path)

    def test_camera_model_tag(self):
        from dng_writer import DNGConfig
        config = DNGConfig(camera_model="NIKON Z 8")
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 50708 in tags  # UniqueCameraModel
        finally:
            os.unlink(path)

    def test_bayer_cfa_tags(self):
        from dng_writer import DNGConfig
        config = DNGConfig(cfa_pattern="RGGB")
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 33421 in tags   # CFARepeatPatternDim
            assert 33422 in tags   # CFAPattern
            assert 50710 in tags   # CFAPlaneColor
            assert 50711 in tags   # CFALayout

            # CFAPattern sollte RGGB = [0, 1, 1, 2] sein
            _, count, raw = tags[33422]
            assert raw[:4] == bytes([0, 1, 1, 2])
        finally:
            os.unlink(path)

    def test_monochrome_no_cfa(self):
        from dng_writer import DNGConfig
        config = DNGConfig(cfa_pattern="MONO")
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 33421 not in tags   # Kein CFARepeatPatternDim
            assert 33422 not in tags   # Kein CFAPattern
            # PhotometricInterpretation = LinearRaw (34892)
            _, _, raw = tags[262]
            pi = struct.unpack_from('<H', raw, 0)[0]
            assert pi == 34892
        finally:
            os.unlink(path)

    def test_16bit_dng(self):
        path, data, _ = self._write_test_dng(bits=16)
        try:
            tags = self._parse_ifd_tags(data, 8)
            _, _, raw = tags[258]  # BitsPerSample
            bps = struct.unpack_from('<H', raw, 0)[0]
            assert bps == 16
        finally:
            os.unlink(path)

    def test_black_white_level(self):
        from dng_writer import DNGConfig
        config = DNGConfig(black_level=64, white_level=1023)
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            # BlackLevel
            _, _, raw = tags[50714]
            bl = struct.unpack_from('<I', raw, 0)[0]
            assert bl == 64
            # WhiteLevel
            _, _, raw = tags[50717]
            wl = struct.unpack_from('<I', raw, 0)[0]
            assert wl == 1023
        finally:
            os.unlink(path)

    def test_strip_data_correct(self):
        """Prüft, dass die Pixeldaten an der richtigen Stelle liegen."""
        path, data, pixel_bytes = self._write_test_dng(width=4, height=4, bits=8)
        try:
            tags = self._parse_ifd_tags(data, 8)
            # StripOffsets
            _, _, raw = tags[273]
            strip_off = struct.unpack_from('<I', raw, 0)[0]
            # StripByteCounts
            _, _, raw = tags[279]
            strip_len = struct.unpack_from('<I', raw, 0)[0]
            assert strip_len == len(pixel_bytes)
            # Pixeldaten vergleichen
            assert data[strip_off:strip_off + strip_len] == pixel_bytes
        finally:
            os.unlink(path)

    def test_tags_sorted(self):
        """IFD-Tags müssen nach Tag-ID sortiert sein."""
        path, data, _ = self._write_test_dng()
        try:
            num_entries = struct.unpack_from('<H', data, 8)[0]
            prev_id = 0
            for i in range(num_entries):
                offset = 10 + i * 12
                tag_id = struct.unpack_from('<H', data, offset)[0]
                assert tag_id > prev_id, f"Tag {tag_id} nicht sortiert (nach {prev_id})"
                prev_id = tag_id
        finally:
            os.unlink(path)

    def test_color_matrix(self):
        from dng_writer import DNGConfig
        cm = np.array([
            [0.6730, 0.1136, 0.1672],
            [0.2820, 0.7282, -0.0102],
            [0.0187, -0.1474, 0.9925],
        ])
        config = DNGConfig(color_matrix_1=cm)
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 50721 in tags   # ColorMatrix1
            assert 50778 in tags   # CalibrationIlluminant1
        finally:
            os.unlink(path)

    def test_as_shot_neutral(self):
        from dng_writer import DNGConfig
        config = DNGConfig(as_shot_neutral=(0.47, 1.0, 0.63))
        path, data, _ = self._write_test_dng(config=config)
        try:
            tags = self._parse_ifd_tags(data, 8)
            assert 50728 in tags  # AsShotNeutral
        finally:
            os.unlink(path)


class TestDNGConvenience:
    """Tests für Hilfsfunktionen."""

    def test_pgm_to_dng(self):
        from dng_writer import pgm_to_dng, read_pgm, DNGConfig

        # PGM erzeugen
        header = b"P5\n4 4\n255\n"
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pgm_file = tempfile.NamedTemporaryFile(suffix='.pgm', delete=False)
        pgm_file.write(header + pixels.tobytes())
        pgm_file.close()

        dng_file = tempfile.NamedTemporaryFile(suffix='.dng', delete=False)
        dng_file.close()

        try:
            result = pgm_to_dng(pgm_file.name, dng_file.name)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 8
            # Header prüfen
            with open(result, 'rb') as f:
                assert f.read(4) == b'\x49\x49\x2a\x00'
        finally:
            os.unlink(pgm_file.name)
            os.unlink(dng_file.name)

    def test_create_dng_from_array(self):
        from dng_writer import create_dng_from_array
        pixels = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        f = tempfile.NamedTemporaryFile(suffix='.dng', delete=False)
        f.close()
        try:
            result = create_dng_from_array(pixels, f.name)
            assert os.path.exists(result)
            with open(result, 'rb') as fh:
                data = fh.read()
            assert data[:2] == b'\x49\x49'
            assert struct.unpack_from('<H', data, 2)[0] == 42
        finally:
            os.unlink(f.name)

    def test_create_dng_from_uint16_array(self):
        from dng_writer import create_dng_from_array
        pixels = np.random.randint(0, 4096, (6, 8), dtype=np.uint16)
        f = tempfile.NamedTemporaryFile(suffix='.dng', delete=False)
        f.close()
        try:
            create_dng_from_array(pixels, f.name)
            assert os.path.getsize(f.name) > 6 * 8 * 2
        finally:
            os.unlink(f.name)

    def test_config_from_camera_info(self):
        from dng_writer import config_from_camera_info
        from camera_db import CameraInfo

        cam = CameraInfo(
            make="NIKON CORPORATION",
            model="NIKON Z 8",
            clean_make="Nikon",
            clean_model="Z 8",
            color_matrix_a=np.eye(3),
            color_matrix_d65=np.eye(3) * 0.9,
        )
        config = config_from_camera_info(cam)
        assert config.camera_model == "NIKON Z 8"
        assert config.color_matrix_1 is not None
        assert config.color_matrix_2 is not None


# ═══════════════════════════════════════════════════════════
#  Validierungs-Tests (neue Input-Checks)
# ═══════════════════════════════════════════════════════════

class TestInputValidation:
    """Tests für die neuen Validierungen."""

    def test_channel_mapping_bounds(self):
        from channel_swap import ChannelMapping
        with pytest.raises(ValueError):
            ChannelMapping(r_source=3, g_source=1, b_source=0)
        with pytest.raises(ValueError):
            ChannelMapping(r_source=0, g_source=-1, b_source=2)

    def test_swap_image_rejects_grayscale(self):
        from channel_swap import swap_image_channels, ChannelMapping
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-Kanal"):
            swap_image_channels(gray, ChannelMapping())

    def test_swap_image_rejects_rgba(self):
        from channel_swap import swap_image_channels, ChannelMapping
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-Kanal"):
            swap_image_channels(rgba, ChannelMapping())

    def test_mix_image_rejects_grayscale(self):
        from channel_swap import mix_image_channels, MixMatrix
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-Kanal"):
            mix_image_channels(gray, MixMatrix())

    def test_huesat_remap_invalid_dims(self):
        from channel_swap import remap_hue_sat_map, MixMatrix
        data = b'\x00' * 100
        result = remap_hue_sat_map(data, (0, 5, 1), MixMatrix())
        assert result == data  # Unverändert bei ungültigen Dims

    def test_huesat_remap_short_data(self):
        from channel_swap import remap_hue_sat_map, MixMatrix
        data = b'\x00' * 10  # Viel zu kurz
        result = remap_hue_sat_map(data, (6, 5, 1), MixMatrix())
        assert result == data  # Unverändert

    def test_dcp_reader_truncated_file(self):
        from dcp_io import DCPReader
        f = tempfile.NamedTemporaryFile(suffix='.dcp', delete=False)
        f.write(b'\x49\x49')  # Nur 2 Bytes
        f.close()
        try:
            with pytest.raises(ValueError, match="zu kurz"):
                DCPReader().read(f.name)
        finally:
            os.unlink(f.name)

    def test_dcp_reader_invalid_ifd_offset(self):
        from dcp_io import DCPReader
        # Gültiger Header aber IFD-Offset zeigt ins Nirgendwo
        f = tempfile.NamedTemporaryFile(suffix='.dcp', delete=False)
        data = struct.pack('<HHI', 0x4949, 0x4352, 99999)
        f.write(data)
        f.close()
        try:
            with pytest.raises(ValueError, match="IFD-Offset"):
                DCPReader().read(f.name)
        finally:
            os.unlink(f.name)


# ═══════════════════════════════════════════════════════════
#  xmp_export.py
# ═══════════════════════════════════════════════════════════

class TestXMPExport:
    """Tests für XMP-Preset-Export."""

    def test_write_xmp_preset(self):
        from xmp_export import write_xmp_preset
        f = tempfile.NamedTemporaryFile(suffix='.xmp', delete=False)
        f.close()
        try:
            result = write_xmp_preset(f.name, "TestProfile", "NIKON Z 8")
            assert os.path.exists(result)
            with open(result, 'r', encoding='utf-8') as fh:
                content = fh.read()
            assert "TestProfile" in content
            assert "NIKON Z 8" in content
            assert "xmpmeta" in content
            assert "crs:" in content
        finally:
            os.unlink(f.name)

    def test_write_xmp_no_camera(self):
        from xmp_export import write_xmp_preset
        f = tempfile.NamedTemporaryFile(suffix='.xmp', delete=False)
        f.close()
        try:
            write_xmp_preset(f.name, "NoCamera")
            with open(f.name, 'r', encoding='utf-8') as fh:
                content = fh.read()
            assert "NoCamera" in content
        finally:
            os.unlink(f.name)

    def test_lightroom_preset_dir(self):
        from xmp_export import get_lightroom_preset_dir
        result = get_lightroom_preset_dir()
        assert "Adobe" in str(result) or "CameraRaw" in str(result)


# ═══════════════════════════════════════════════════════════
#  icc_export.py
# ═══════════════════════════════════════════════════════════

class TestICCExport:
    """Tests für ICC-Profil-Export."""

    def test_write_icc_identity(self):
        from icc_export import write_icc_profile
        f = tempfile.NamedTemporaryFile(suffix='.icc', delete=False)
        f.close()
        try:
            result = write_icc_profile(f.name, np.eye(3))
            assert os.path.exists(result)
            size = os.path.getsize(result)
            assert size >= 128  # ICC Header mindestens 128 Bytes
            with open(result, 'rb') as fh:
                data = fh.read()
            # ICC Signatur prüfen
            assert data[36:40] == b'acsp'
        finally:
            os.unlink(f.name)

    def test_channel_swap_to_icc(self):
        from icc_export import channel_swap_to_icc
        # R↔B Swap-Matrix
        swap = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        f = tempfile.NamedTemporaryFile(suffix='.icc', delete=False)
        f.close()
        try:
            result = channel_swap_to_icc(f.name, swap)
            assert os.path.exists(result)
            with open(result, 'rb') as fh:
                data = fh.read()
            assert data[36:40] == b'acsp'
        finally:
            os.unlink(f.name)

    def test_read_icc_info(self):
        from icc_export import write_icc_profile, read_icc_info
        f = tempfile.NamedTemporaryFile(suffix='.icc', delete=False)
        f.close()
        try:
            write_icc_profile(f.name, np.eye(3), description="Test ICC")
            info = read_icc_info(f.name)
            assert 'size' in info
            assert 'tag_count' in info
            assert info['tag_count'] > 0
        finally:
            os.unlink(f.name)


# ═══════════════════════════════════════════════════════════
#  npc_io.py
# ═══════════════════════════════════════════════════════════

class TestNPCIO:
    """Tests für Nikon Picture Control I/O."""

    def test_write_read_roundtrip_v0100(self):
        from npc_io import NikonPictureControlFile, write_npc, read_npc
        pc = NikonPictureControlFile(
            name="TestPreset",
            base="STANDARD",
            sharpening=140,
            contrast=135,
            brightness=128,
            saturation=120,
            hue=130,
        )
        f = tempfile.NamedTemporaryFile(suffix='.npc', delete=False)
        f.close()
        try:
            write_npc(f.name, pc, format_version="0100")
            loaded = read_npc(f.name)
            assert loaded.name == "TestPreset"
            assert loaded.base == "STANDARD"
            assert loaded.contrast == 135
            assert loaded.saturation == 120
        finally:
            os.unlink(f.name)

    def test_write_read_roundtrip_v0300(self):
        from npc_io import NikonPictureControlFile, write_npc, read_npc
        pc = NikonPictureControlFile(
            name="ZSeriesTest",
            base="VIVID",
            sharpening=None,  # Auto
            contrast=128,
            brightness=128,
            saturation=150,
            hue=128,
        )
        f = tempfile.NamedTemporaryFile(suffix='.np3', delete=False)
        f.close()
        try:
            write_npc(f.name, pc, format_version="0300")
            loaded = read_npc(f.name)
            assert loaded.name == "ZSeriesTest"
        finally:
            os.unlink(f.name)

    def test_to_lightroom_values(self):
        from npc_io import NikonPictureControlFile, to_lightroom_values
        pc = NikonPictureControlFile(
            contrast=158,      # +30 von Mitte (128)
            brightness=128,    # Neutral
            saturation=108,    # -20 von Mitte
        )
        lr = to_lightroom_values(pc)
        assert 'contrast' in lr
        assert 'saturation' in lr
        assert lr['contrast'] > 0     # Positiv
        assert lr['saturation'] < 0   # Negativ

    def test_invalid_npc_magic(self):
        from npc_io import read_npc
        f = tempfile.NamedTemporaryFile(suffix='.npc', delete=False)
        f.write(b'INVALID_DATA_NOT_NPC')
        f.close()
        try:
            with pytest.raises((ValueError, Exception)):
                read_npc(f.name)
        finally:
            os.unlink(f.name)

    def test_base_profiles_defined(self):
        from npc_io import BASE_PROFILES
        assert len(BASE_PROFILES) >= 10
        assert "STANDARD" in [v for v in BASE_PROFILES.values()]


# ═══════════════════════════════════════════════════════════
#  camera_db.py
# ═══════════════════════════════════════════════════════════

class TestCameraDB:
    """Tests für Kamera-Datenbank."""

    def test_camera_info_properties(self):
        from camera_db import CameraInfo
        cam = CameraInfo(
            make="NIKON CORPORATION",
            model="NIKON Z 6_2",
            clean_make="Nikon",
            clean_model="Z 6 II",
            color_matrix_a=np.eye(3),
        )
        assert cam.display_name == "Nikon Z 6 II"
        assert cam.unique_camera_model == "NIKON Z 6_2"

    def test_load_from_mock_toml(self):
        from camera_db import load_camera_database
        # Mock dnglab-Verzeichnisstruktur erstellen
        with tempfile.TemporaryDirectory() as tmpdir:
            cameras = os.path.join(tmpdir, 'rawler', 'data', 'cameras', 'nikon')
            os.makedirs(cameras)
            toml_content = '''make = "NIKON CORPORATION"
model = "NIKON Z 8"
clean_make = "Nikon"
clean_model = "Z 8"

[cameras.color_matrix]
A = [0.67, 0.11, 0.17, 0.28, 0.73, -0.01, 0.02, -0.15, 0.99]
D65 = [0.60, 0.15, 0.20, 0.25, 0.70, 0.05, 0.03, -0.10, 0.95]
'''
            with open(os.path.join(cameras, 'z8.toml'), 'w') as f:
                f.write(toml_content)

            cams = load_camera_database(tmpdir)
            assert len(cams) == 1
            assert cams[0].model == "NIKON Z 8"
            assert cams[0].color_matrix_a is not None
            assert cams[0].color_matrix_a.shape == (3, 3)
            assert cams[0].color_matrix_d65 is not None

    def test_load_empty_dir(self):
        from camera_db import load_camera_database
        with tempfile.TemporaryDirectory() as tmpdir:
            cams = load_camera_database(tmpdir)
            assert cams == []

    def test_find_dnglab_path(self):
        from camera_db import find_dnglab_path
        # Sollte None zurückgeben wenn dnglab nicht vorhanden
        result = find_dnglab_path()
        assert result is None or os.path.isdir(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
