#!/usr/bin/env python3
"""
DNG Channel Tool - Farbkanal-Tausch & Adobe DCP-Export

Features:
- Bild-Vorschau mit Zoom/Pan und Vorher/Nachher-Split
- RGB-Histogramm
- Alle Kanal-Permutationen + gewichteter Kanal-Mix
- IR-Fotografie-Presets
- DCP-Profil-Export mit HueSatMap-Remapping
- XMP-Preset-Export für Lightroom
- Auto-Kamera-Erkennung aus EXIF
- Batch-Verarbeitung
- Adobe-Profil-Browser
- Undo/Redo für alle Bildoperationen

Modulstruktur:
- main.py: Hauptanwendung (ChannelToolApp) + Einstiegspunkt
- gui_widgets.py: Wiederverwendbare UI-Komponenten
- gui_dialogs.py: Alle Dialog-Fenster
- undo.py: Undo/Redo-Verwaltung
- logging_setup.py: Logging-Konfiguration
"""

import os
import sys
import json
import logging
import argparse
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List
import numpy as np
from PIL import Image, ImageTk

from logging_setup import setup_logging
from undo import UndoManager
from gui_widgets import AutocompleteCombobox, HistogramWidget
from gui_dialogs import (
    BatchDialog, NEFExtractDialog, FujiRecipeDialog,
    StyleResultDialog, NikonPresetCreatorDialog, PresetLibraryDialog,
)

from channel_swap import (
    ChannelMapping, MixMatrix, SWAP_PRESETS, CHANNEL_SHORT, IR_PRESETS,
    swap_image_channels, apply_to_image,
    swap_color_matrix, swap_forward_matrix,
    apply_to_color_matrix, apply_to_forward_matrix,
    remap_hue_sat_map,
)
from dcp_io import (
    DCPProfile, DCPWriter, DCPReader,
    ILLUMINANT_A, ILLUMINANT_D65, ILLUMINANT_NAMES,
    install_dcp_to_adobe, get_adobe_profile_dir,
    rewrite_dcp_camera_model,
)
from camera_db import load_camera_database, find_dnglab_path, CameraInfo

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    from xmp_export import write_xmp_preset, get_lightroom_preset_dir, install_xmp_to_lightroom
    HAS_XMP = True
except ImportError:
    HAS_XMP = False

try:
    from nef_extract import (
        extract_picture_control, picture_control_to_xmp,
        save_preview, print_picture_control, NikonPictureControl,
    )
    HAS_NEF_EXTRACT = True
except ImportError:
    HAS_NEF_EXTRACT = False

try:
    from npc_io import (
        NikonPictureControlFile, read_npc, write_npc,
        from_nef_picture_control, to_lightroom_values,
        find_sd_cards, install_to_camera, BASE_PROFILES,
    )
    HAS_NPC = True
except ImportError:
    HAS_NPC = False

try:
    from style_transfer import (
        analyze_image, compare_images, apply_style,
        style_to_xmp, style_to_nikon_pc, ImageStyle,
    )
    HAS_STYLE = True
except ImportError:
    HAS_STYLE = False

try:
    from fuji_recipe import (
        FujiRecipe, parse_recipe, recipe_to_xmp, recipe_to_nikon_pc,
        install_recipe_to_lightroom, EXAMPLE_RECIPES, FILM_SIM_TO_ADOBE,
    )
    HAS_FUJI = True
except ImportError:
    HAS_FUJI = False

try:
    from ir_tools import (
        IR_FILTERS, IR_FALSE_COLOR_PRESETS, IRFalseColorPreset,
        calculate_ir_wb, apply_ir_wb, apply_ir_preset,
        simulate_ir_filter, detect_hotspot, correct_hotspot,
        generate_ir_dcp, calculate_ndvi, ndvi_statistics,
        ir_preset_to_xmp,
    )
    HAS_IR = True
except ImportError:
    HAS_IR = False

try:
    from lut_export import (
        write_cube_lut, mix_matrix_to_lut, tone_curve_to_lut,
        style_to_lut, combined_lut, fuji_recipe_to_lut,
    )
    HAS_LUT = True
except ImportError:
    HAS_LUT = False

try:
    from wb_picker import (
        calculate_wb_from_pixel, apply_wb_correction,
        extract_camera_jpeg, compare_jpeg_vs_raw, histogram_match,
    )
    HAS_WB = True
except ImportError:
    HAS_WB = False

try:
    from color_checker import (
        calibrate_from_colorchecker, calibration_to_dcp, calibration_to_lut,
        sample_patch_color, COLORCHECKER_NAMES,
    )
    HAS_CHECKER = True
except ImportError:
    HAS_CHECKER = False

try:
    from preset_library import scan_all_presets, filter_presets, get_preset_info
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False

try:
    from camera_presets import (
        CanonPictureStyle, SonyCreativeLook,
        canon_to_xmp, sony_to_xmp, canon_to_nikon, sony_to_nikon,
        CANON_BASE_STYLES, SONY_BASE_LOOKS,
    )
    HAS_CANON_SONY = True
except ImportError:
    HAS_CANON_SONY = False

try:
    from icc_export import write_icc_profile, channel_swap_to_icc
    HAS_ICC = True
except ImportError:
    HAS_ICC = False

try:
    from dng_writer import (
        read_pgm, DNGWriter, DNGConfig, PGMData,
        pgm_to_dng, create_dng_from_array, config_from_camera_info,
        CFA_PATTERNS,
    )
    HAS_DNG_WRITER = True
except ImportError:
    HAS_DNG_WRITER = False

try:
    from dcp_xml import (
        dcp_to_xml, xml_to_dcp, export_dcp_to_xml, import_xml_to_dcp,
        make_invariant, untwist,
    )
    HAS_DCP_XML = True
except ImportError:
    HAS_DCP_XML = False


logger = logging.getLogger(__name__)

WINDOW_TITLE = "DNG Channel Tool - Farbkanal-Tausch & Adobe DCP-Export"

# ── Config / Recent Files ────────────────────────────────────
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".dng_channel_tool")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
MAX_RECENT = 10


def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(data: dict):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ── Main Application ──────────────────────────────────────────

class ChannelToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x860")
        self.root.minsize(900, 600)

        # ── State ──
        self.original_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.photo_image_split: Optional[ImageTk.PhotoImage] = None
        self.current_file: Optional[str] = None
        self.cameras: list = []
        self.camera_map: dict = {}
        self.loaded_dcp: Optional[DCPProfile] = None

        # Channel mapping (permutation mode)
        self.r_var = tk.IntVar(value=0)
        self.g_var = tk.IntVar(value=1)
        self.b_var = tk.IntVar(value=2)

        # Mix mode
        self.mix_mode = tk.BooleanVar(value=False)
        self.mix_vars = [[tk.DoubleVar(value=1.0 if i == j else 0.0)
                          for j in range(3)] for i in range(3)]

        # View state
        self.split_view = tk.BooleanVar(value=False)
        self.show_histogram = tk.BooleanVar(value=True)
        self.zoom_level = 1.0  # 1.0 = fit
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start = None
        self.wb_picker_active = False  # WB-Pipette Modus

        # Detected camera from EXIF
        self.detected_camera: Optional[str] = None

        # Recent files
        self.recent_files: List[str] = _load_config().get("recent_files", [])[:MAX_RECENT]

        # Undo/Redo
        self.undo_manager = UndoManager()
        self.undo_manager.on_change(self._update_undo_menu_state)

        # Load camera database
        self._load_cameras()

        # Build UI
        self._build_menu()
        self._build_ui()
        self._setup_dnd()  # Fix #1
        self._update_status("Bereit – Bild oder DCP-Profil laden zum Starten.")

    def _load_cameras(self):
        dnglab_path = find_dnglab_path()
        if dnglab_path:
            self.cameras = load_camera_database(dnglab_path)
            self.camera_map = {cam.display_name: cam for cam in self.cameras}

    # ── Menu ──────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # Datei
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Bild öffnen…", command=self._open_image,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="DCP-Profil laden…", command=self._open_dcp)
        file_menu.add_command(label="Adobe-Profile durchsuchen…",
                              command=self._browse_adobe_profiles)  # Feature #10
        file_menu.add_separator()
        file_menu.add_command(label="Bild speichern…", command=self._save_image,
                              accelerator="Ctrl+S")
        file_menu.add_command(label="DCP exportieren…", command=self._export_dcp,
                              accelerator="Ctrl+E")
        if HAS_XMP:
            file_menu.add_command(label="XMP-Preset exportieren…",
                                  command=self._export_xmp)  # Feature #14
        file_menu.add_command(label="DCP in Adobe installieren",
                              command=self._install_to_adobe)
        if HAS_LUT:
            file_menu.add_command(label="3D LUT exportieren (.cube)…",
                                  command=self._export_lut)
        if HAS_ICC:
            file_menu.add_command(label="ICC-Profil exportieren…",
                                  command=self._export_icc)
        file_menu.add_command(label="Alles exportieren…",
                              command=self._export_all,
                              accelerator="Ctrl+Shift+E")
        file_menu.add_separator()
        file_menu.add_command(label="Batch-Verarbeitung…",
                              command=self._open_batch_dialog,
                              accelerator="Ctrl+B")
        file_menu.add_separator()

        # Zuletzt geöffnet
        self._recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Zuletzt geöffnet", menu=self._recent_menu)
        self._update_recent_menu()

        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        menubar.add_cascade(label="Datei", menu=file_menu)

        # Bearbeiten (Undo/Redo)
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Rückgängig", command=self._undo,
                              accelerator="Ctrl+Z", state='disabled')
        edit_menu.add_command(label="Wiederherstellen", command=self._redo,
                              accelerator="Ctrl+Y", state='disabled')
        self._edit_menu = edit_menu
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)

        # Ansicht
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Vorher/Nachher Split",
                                  variable=self.split_view,
                                  command=self._on_view_changed,
                                  accelerator="V")  # Feature #5
        view_menu.add_checkbutton(label="Histogramm",
                                  variable=self.show_histogram,
                                  command=self._on_histogram_toggle)  # Feature #6
        view_menu.add_separator()
        view_menu.add_command(label="Zoom zurücksetzen",
                              command=self._reset_zoom, accelerator="Doppelklick")
        menubar.add_cascade(label="Ansicht", menu=view_menu)

        # Extras
        extras_menu = tk.Menu(menubar, tearoff=0)
        if HAS_NEF_EXTRACT:
            extras_menu.add_command(
                label="NEF Picture Control extrahieren…",
                command=self._extract_nef_picture_control,
                accelerator="Ctrl+N")
            extras_menu.add_command(
                label="NEF → Lightroom-Preset (Schnell)…",
                command=self._quick_nef_to_preset)
        if HAS_NPC:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="Nikon Preset erstellen…",
                command=self._create_nikon_preset)
            extras_menu.add_command(
                label="NPC/NP3-Datei laden…",
                command=self._open_npc_file)
            extras_menu.add_command(
                label="NPC/NP3 auf SD-Karte kopieren…",
                command=self._install_npc_to_card)
        if HAS_STYLE:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="Bildstil analysieren…",
                command=self._analyze_style)
            extras_menu.add_command(
                label="Stil von Referenzbild übertragen…",
                command=self._transfer_style)
            extras_menu.add_command(
                label="Original + Bearbeitet vergleichen → Preset…",
                command=self._compare_style)
        if HAS_FUJI:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="Fujifilm-Rezept eingeben / konvertieren…",
                command=self._open_fuji_recipe_dialog)
        if HAS_CANON_SONY:
            extras_menu.add_command(
                label="Canon/Sony Preset erstellen…",
                command=self._create_canon_sony_preset)
        if HAS_WB:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="Weißabgleich-Pipette aktivieren",
                command=self._toggle_wb_picker,
                accelerator="W")
            extras_menu.add_command(
                label="Histogram Matching (Referenzbild)…",
                command=self._histogram_match)
            extras_menu.add_command(
                label="Kamera-JPEG vs. RAW vergleichen…",
                command=self._compare_jpeg_raw)
        if HAS_CHECKER:
            extras_menu.add_command(
                label="Color Checker Kalibrierung…",
                command=self._color_checker_calibrate)
        if HAS_LIBRARY:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="Preset-Bibliothek öffnen…",
                command=self._open_preset_library,
                accelerator="Ctrl+L")
        extras_menu.add_separator()
        extras_menu.add_command(
            label="Aktuelles Bild: Kamera-Info anzeigen",
            command=self._show_camera_info)
        extras_menu.add_command(
            label="DCP-Kameramodell umschreiben…",
            command=self._rewrite_dcp_model)
        if HAS_DNG_WRITER:
            extras_menu.add_command(
                label="PGM → DNG konvertieren…",
                command=self._pgm_to_dng)
        if HAS_DCP_XML:
            extras_menu.add_separator()
            extras_menu.add_command(
                label="DCP → XML exportieren…",
                command=self._export_dcp_xml)
            extras_menu.add_command(
                label="XML → DCP kompilieren…",
                command=self._import_xml_dcp)
            extras_menu.add_command(
                label="DCP invariant machen…",
                command=self._dcp_make_invariant)
            extras_menu.add_command(
                label="DCP untwisten…",
                command=self._dcp_untwist)
        menubar.add_cascade(label="Extras", menu=extras_menu)

        # Infrarot
        if HAS_IR:
            ir_menu = tk.Menu(menubar, tearoff=0)
            ir_menu.add_command(label="IR-Weißabgleich (Klick auf Vegetation)…",
                                command=self._ir_wb_picker)
            ir_menu.add_separator()

            # False Color Presets Submenu
            fc_menu = tk.Menu(ir_menu, tearoff=0)
            for preset_name in IR_FALSE_COLOR_PRESETS:
                fc_menu.add_command(
                    label=preset_name,
                    command=lambda n=preset_name: self._apply_ir_false_color(n))
            ir_menu.add_cascade(label="False-Color Presets", menu=fc_menu)

            # Filter simulation submenu
            sim_menu = tk.Menu(ir_menu, tearoff=0)
            for filter_name in IR_FILTERS:
                sim_menu.add_command(
                    label=f"{filter_name} - {IR_FILTERS[filter_name].name}",
                    command=lambda f=filter_name: self._simulate_ir_filter(f))
            ir_menu.add_cascade(label="Filter-Simulation", menu=sim_menu)

            ir_menu.add_separator()
            ir_menu.add_command(label="Hotspot-Erkennung…",
                                command=self._detect_hotspot)
            ir_menu.add_command(label="NDVI berechnen (Vegetationsindex)…",
                                command=self._calculate_ndvi)
            ir_menu.add_separator()
            ir_menu.add_command(label="IR-DCP-Profil generieren…",
                                command=self._generate_ir_dcp)
            ir_menu.add_command(label="IR-Preset als LUT/XMP exportieren…",
                                command=self._export_ir_preset)
            menubar.add_cascade(label="Infrarot", menu=ir_menu)

        self.root.config(menu=menubar)

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-s>', lambda e: self._save_image())
        self.root.bind('<Control-e>', lambda e: self._export_dcp())
        self.root.bind('<Control-b>', lambda e: self._open_batch_dialog())
        self.root.bind('<Control-n>', lambda e: self._extract_nef_picture_control()
                       if HAS_NEF_EXTRACT else None)
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
        self.root.bind('<v>', lambda e: self._toggle_split())
        self.root.bind('<V>', lambda e: self._toggle_split())
        self.root.bind('<w>', lambda e: self._toggle_wb_picker() if HAS_WB else None)
        self.root.bind('<Control-l>', lambda e: self._open_preset_library()
                       if HAS_LIBRARY else None)
        self.root.bind('<Control-Shift-E>', lambda e: self._export_all())

    # ── UI Building ───────────────────────────────────────────

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left: Preview + Histogram ──
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        # Canvas with zoom/pan (#7)
        self.canvas = tk.Canvas(left_frame, bg='#2b2b2b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)  # Zoom #7
        self.canvas.bind('<ButtonPress-1>', self._on_pan_start)  # Pan #7
        self.canvas.bind('<B1-Motion>', self._on_pan_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_pan_end)
        self.canvas.bind('<Double-Button-1>', lambda e: self._reset_zoom())

        # Histogram (#6)
        self.histogram = HistogramWidget(left_frame, height=80)
        self.histogram.pack(fill=tk.X)

        # ── Right: Controls ──
        right_frame = ttk.Frame(main_pane, width=370)
        main_pane.add(right_frame, weight=1)

        canvas_scroll = tk.Canvas(right_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas_scroll.yview)
        scroll_frame = ttk.Frame(canvas_scroll)
        scroll_frame.bind('<Configure>',
                          lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox('all')))
        canvas_scroll.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling for right panel
        def _on_scroll_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas_scroll.bind_all('<MouseWheel>', lambda e: None)  # Will rebind per-widget
        scroll_frame.bind('<Enter>',
                          lambda e: canvas_scroll.bind_all('<MouseWheel>', _on_scroll_mousewheel))
        scroll_frame.bind('<Leave>',
                          lambda e: canvas_scroll.bind_all('<MouseWheel>', lambda ev: None))

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        pad = {'padx': 8, 'pady': 3}

        # ── Mode Toggle ──
        mode_frame = ttk.LabelFrame(scroll_frame, text="  Modus  ", padding=8)
        mode_frame.pack(fill=tk.X, **pad)

        ttk.Radiobutton(mode_frame, text="Kanal-Tausch (Permutation)",
                         variable=self.mix_mode, value=False,
                         command=self._on_mode_changed).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Gewichteter Mix (3×3 Matrix)",
                         variable=self.mix_mode, value=True,
                         command=self._on_mode_changed).pack(anchor='w')

        # ── Permutation Section ──
        self.perm_frame = ttk.LabelFrame(scroll_frame, text="  Kanal-Zuordnung  ", padding=8)
        self.perm_frame.pack(fill=tk.X, **pad)

        channels = ["R (Rot)", "G (Grün)", "B (Blau)"]
        colors = ['#ff4444', '#44cc44', '#4488ff']

        for i, (label, var, color) in enumerate([
            ("Ausgang R  ←", self.r_var, colors[0]),
            ("Ausgang G  ←", self.g_var, colors[1]),
            ("Ausgang B  ←", self.b_var, colors[2]),
        ]):
            row = ttk.Frame(self.perm_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, fg=color, font=('Consolas', 11, 'bold')).pack(side=tk.LEFT)
            combo = ttk.Combobox(row, values=channels, state='readonly', width=12)
            combo.pack(side=tk.RIGHT)
            combo.current(var.get())
            combo.bind('<<ComboboxSelected>>',
                       lambda e, v=var, c=combo: self._on_channel_combo(v, c))
            setattr(self, f'combo_{CHANNEL_SHORT[i].lower()}', combo)

        # ── Swap Presets ──
        preset_frame = ttk.LabelFrame(scroll_frame, text="  Schnell-Presets  ", padding=8)
        preset_frame.pack(fill=tk.X, **pad)

        btn_grid = ttk.Frame(preset_frame)
        btn_grid.pack(fill=tk.X)
        for i, (name, perm) in enumerate(SWAP_PRESETS.items()):
            btn = ttk.Button(btn_grid, text=name, width=16,
                             command=lambda p=perm: self._apply_preset(p))
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky='ew')
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # ── IR Presets (#12) ──
        ir_frame = ttk.LabelFrame(scroll_frame, text="  IR-Presets  ", padding=8)
        ir_frame.pack(fill=tk.X, **pad)

        ir_grid = ttk.Frame(ir_frame)
        ir_grid.pack(fill=tk.X)
        for i, (name, matrix) in enumerate(IR_PRESETS.items()):
            btn = ttk.Button(ir_grid, text=name,
                             command=lambda m=matrix, n=name: self._apply_ir_preset(m, n))
            btn.grid(row=i, column=0, padx=2, pady=1, sticky='ew')
        ir_grid.columnconfigure(0, weight=1)

        # ── Mix Matrix Section (#11) ──
        self.mix_frame = ttk.LabelFrame(scroll_frame, text="  Mix-Matrix (Gewichtung %)  ", padding=8)
        self.mix_frame.pack(fill=tk.X, **pad)

        header = ttk.Frame(self.mix_frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text="", width=10).pack(side=tk.LEFT)
        for ch, color in zip(["R ein", "G ein", "B ein"], colors):
            tk.Label(header, text=ch, fg=color, font=('Consolas', 9, 'bold'),
                     width=8).pack(side=tk.LEFT, padx=2)

        self.mix_scales = []
        for out_ch in range(3):
            row = ttk.Frame(self.mix_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{CHANNEL_SHORT[out_ch]} aus:", fg=colors[out_ch],
                     font=('Consolas', 9, 'bold'), width=10).pack(side=tk.LEFT)
            row_scales = []
            for in_ch in range(3):
                var = self.mix_vars[out_ch][in_ch]
                scale = ttk.Scale(row, from_=0, to=1.0, variable=var,
                                  orient=tk.HORIZONTAL, length=80,
                                  command=lambda v, o=out_ch, i=in_ch: self._on_mix_changed())
                scale.pack(side=tk.LEFT, padx=2)
                row_scales.append(scale)
            self.mix_scales.append(row_scales)

        norm_btn = ttk.Button(self.mix_frame, text="Normalisieren (Zeilen → 100%)",
                              command=self._normalize_mix)
        norm_btn.pack(fill=tk.X, pady=(5, 0))

        # ── Matrix Preview ──
        mat_frame = ttk.LabelFrame(scroll_frame, text="  Matrix-Vorschau  ", padding=8)
        mat_frame.pack(fill=tk.X, **pad)
        self.matrix_label = tk.Label(mat_frame, font=('Consolas', 9),
                                     justify=tk.LEFT, anchor='w')
        self.matrix_label.pack(fill=tk.X)

        # ── DCP Settings ──
        dcp_frame = ttk.LabelFrame(scroll_frame, text="  DCP-Profil Einstellungen  ", padding=8)
        dcp_frame.pack(fill=tk.X, **pad)

        ttk.Label(dcp_frame, text="Kamera-Modell (Suche mit Tippen):").pack(anchor='w')
        self.camera_var = tk.StringVar()
        self.camera_combo = AutocompleteCombobox(dcp_frame, textvariable=self.camera_var,
                                                  width=35)  # Fix #4
        if self.cameras:
            self.camera_combo.set_all_values(sorted(self.camera_map.keys()))
        self.camera_combo.pack(fill=tk.X, pady=(0, 5))
        self.camera_combo.bind('<<ComboboxSelected>>', self._on_camera_selected)

        ttk.Label(dcp_frame, text="Oder Kameraname manuell eingeben:").pack(anchor='w')
        self.manual_camera_var = tk.StringVar()
        ttk.Entry(dcp_frame, textvariable=self.manual_camera_var, width=35).pack(fill=tk.X, pady=(0, 5))

        # Auto-detected camera info (#8)
        self.detected_label = ttk.Label(dcp_frame, text="", foreground='#888888')
        self.detected_label.pack(anchor='w')

        ttk.Label(dcp_frame, text="Profilname:").pack(anchor='w')
        self.profile_name_var = tk.StringVar(value="Channel Swap")
        ttk.Entry(dcp_frame, textvariable=self.profile_name_var, width=35).pack(fill=tk.X, pady=(0, 5))

        # ── Export Buttons ──
        export_frame = ttk.LabelFrame(scroll_frame, text="  Export  ", padding=8)
        export_frame.pack(fill=tk.X, **pad)

        ttk.Button(export_frame, text="DCP-Profil exportieren…",
                   command=self._export_dcp).pack(fill=tk.X, pady=2)
        if HAS_XMP:
            ttk.Button(export_frame, text="XMP-Preset exportieren…",
                       command=self._export_xmp).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="DCP in Adobe installieren",
                   command=self._install_to_adobe).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Bild speichern…",
                   command=self._save_image).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Batch-Verarbeitung…",
                   command=self._open_batch_dialog).pack(fill=tk.X, pady=2)

        info_text = (
            "DCP-Profile werden von Adobe Lightroom\n"
            "und Camera Raw als Kamera-Profil erkannt.\n\n"
            f"Adobe-Ordner:\n{get_adobe_profile_dir()}"
        )
        ttk.Label(export_frame, text=info_text, wraplength=280,
                  foreground='gray', justify=tk.LEFT).pack(anchor='w', pady=(8, 0))

        # ── Status Bar ──
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2)).pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(status_frame, textvariable=self.zoom_var,
                  relief=tk.SUNKEN, anchor=tk.CENTER, padding=(5, 2), width=10).pack(side=tk.RIGHT)

        # Initial display
        self._on_mode_changed()
        self._update_matrix_display()

    # ── Mode / Mapping ────────────────────────────────────────

    def _on_mode_changed(self):
        """Toggle between permutation and mix mode."""
        is_mix = self.mix_mode.get()
        # Show/hide sections visually (disable widgets)
        for child in self.perm_frame.winfo_children():
            try:
                child.configure(state='disabled' if is_mix else 'normal')
            except tk.TclError:
                pass
        for row in self.mix_scales:
            for scale in row:
                scale.configure(state='!disabled' if is_mix else 'disabled')

        self._on_mapping_changed()

    def _get_mix_matrix(self) -> MixMatrix:
        """Get the current mix matrix from UI state."""
        if self.mix_mode.get():
            m = np.array([[self.mix_vars[i][j].get() for j in range(3)] for i in range(3)])
            return MixMatrix(matrix=m)
        else:
            mapping = ChannelMapping(
                r_source=self.r_var.get(),
                g_source=self.g_var.get(),
                b_source=self.b_var.get(),
            )
            return mapping.to_mix_matrix()

    def _on_channel_combo(self, var: tk.IntVar, combo: ttk.Combobox):
        var.set(combo.current())
        self._on_mapping_changed()

    def _apply_preset(self, perm):
        self._push_undo("Kanal-Tausch")
        self.mix_mode.set(False)
        self.r_var.set(perm[0])
        self.g_var.set(perm[1])
        self.b_var.set(perm[2])
        self.combo_r.current(perm[0])
        self.combo_g.current(perm[1])
        self.combo_b.current(perm[2])
        self._on_mode_changed()

    def _apply_ir_preset(self, matrix: np.ndarray, name: str):
        """Apply an IR preset mix matrix (#12)."""
        self._push_undo(f"IR: {name}")
        self.mix_mode.set(True)
        for i in range(3):
            for j in range(3):
                self.mix_vars[i][j].set(float(matrix[i, j]))
        self.profile_name_var.set(name)
        self._on_mode_changed()

    def _on_mix_changed(self):
        self._on_mapping_changed()

    def _normalize_mix(self):
        """Normalize each row of the mix matrix to sum to 1.0."""
        for i in range(3):
            total = sum(self.mix_vars[i][j].get() for j in range(3))
            if total > 0:
                for j in range(3):
                    self.mix_vars[i][j].set(self.mix_vars[i][j].get() / total)
        self._on_mapping_changed()

    def _on_mapping_changed(self, *args):
        mix = self._get_mix_matrix()

        # Update profile name
        cam_name = self.camera_var.get()
        if cam_name and cam_name in self.camera_map:
            cam = self.camera_map[cam_name]
            self.profile_name_var.set(f"{cam.clean_make} {cam.clean_model} {mix.name}")

        # Update image preview
        if self.original_image is not None:
            self.preview_image = apply_to_image(self.original_image, mix)
            self._display_preview()
            self._update_histogram()

        self._update_matrix_display()

    # ── File Operations ───────────────────────────────────────

    def _open_image(self):
        filetypes = [("Alle Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        if HAS_RAWPY:
            filetypes.insert(0, ("RAW/DNG", "*.dng *.cr2 *.cr3 *.nef *.arw *.orf *.rw2 *.raf *.pef *.srw"))
            filetypes.insert(0, ("Alle unterstützten",
                                 "*.dng *.cr2 *.cr3 *.nef *.arw *.orf *.rw2 *.raf *.pef *.srw "
                                 "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"))

        path = filedialog.askopenfilename(title="Bild öffnen", filetypes=filetypes)
        if path:
            self._load_image_threaded(path)  # Fix #3

    def _load_image_threaded(self, path: str):
        """Load image in background thread to keep GUI responsive (Fix #3)."""
        self._update_status(f"Lade: {os.path.basename(path)}…")
        self.root.update()

        def _load():
            try:
                ext = os.path.splitext(path)[1].lower()
                raw_exts = {'.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.raf', '.pef', '.srw'}

                camera_model = None

                if ext in raw_exts and HAS_RAWPY:
                    with rawpy.imread(path) as raw:
                        image = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
                        # Auto camera detection (#8)
                        try:
                            camera_model = raw.camera_model
                        except Exception:
                            pass
                else:
                    img = Image.open(path).convert('RGB')
                    image = np.array(img)
                    # EXIF camera detection (#8)
                    try:
                        exif = img.getexif()
                        if exif:
                            camera_model = exif.get(0x0110, '')  # Model tag
                    except Exception:
                        pass

                # Update UI in main thread
                self.root.after(0, lambda: self._on_image_loaded(path, image, camera_model))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Fehler", f"Datei konnte nicht geladen werden:\n{e}"))
                self.root.after(0, lambda: self._update_status("Fehler beim Laden."))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def _on_image_loaded(self, path: str, image: np.ndarray, camera_model: Optional[str]):
        """Called after image loading completes (main thread)."""
        self.original_image = image
        self.current_file = path
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.undo_manager.clear()
        self._add_to_recent(path)
        logger.info("Bild geladen: %s (%dx%d)", os.path.basename(path), image.shape[1], image.shape[0])

        # Auto camera detection (#8)
        if camera_model:
            self.detected_camera = camera_model
            self.detected_label.config(text=f"Erkannt: {camera_model}")

            # Try to match in database
            for name, cam in self.camera_map.items():
                if cam.model == camera_model or cam.clean_model == camera_model:
                    self.camera_var.set(name)
                    self.manual_camera_var.set(cam.unique_camera_model)
                    break
            else:
                self.manual_camera_var.set(camera_model)
        else:
            self.detected_camera = None
            self.detected_label.config(text="")

        self._on_mapping_changed()
        self._update_status(
            f"Geladen: {os.path.basename(path)} "
            f"({image.shape[1]}×{image.shape[0]})"
        )

    def _open_dcp(self, path: str = None):
        """Open an existing DCP profile (Fix #2 - accepts path parameter)."""
        if path is None:
            path = filedialog.askopenfilename(
                title="DCP-Profil laden",
                filetypes=[("DNG Camera Profile", "*.dcp"), ("Alle Dateien", "*.*")]
            )
        if not path:
            return

        try:
            self.loaded_dcp = DCPReader().read(path)
            self.manual_camera_var.set(self.loaded_dcp.camera_model)
            self.profile_name_var.set(self.loaded_dcp.profile_name + " (Swap)")
            self._update_matrix_display()
            self._add_to_recent(path)
            self._update_status(
                f"DCP geladen: {self.loaded_dcp.profile_name} "
                f"({self.loaded_dcp.camera_model})")
        except Exception as e:
            messagebox.showerror("Fehler", f"DCP konnte nicht geladen werden:\n{e}")

    def _browse_adobe_profiles(self):
        """Browse existing Adobe DCP profiles (Feature #10)."""
        adobe_dir = get_adobe_profile_dir()
        if not os.path.isdir(adobe_dir):
            messagebox.showinfo("Info", f"Adobe-Profilordner nicht gefunden:\n{adobe_dir}")
            return

        path = filedialog.askopenfilename(
            title="Adobe-Profil auswählen",
            initialdir=adobe_dir,
            filetypes=[("DNG Camera Profile", "*.dcp"), ("Alle Dateien", "*.*")]
        )
        if path:
            self._open_dcp(path)

    def _save_image(self):
        if self.preview_image is None:
            messagebox.showinfo("Info", "Kein Bild geladen.")
            return

        path = filedialog.asksaveasfilename(
            title="Bild speichern", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")])
        if not path:
            return

        try:
            Image.fromarray(self.preview_image).save(path, quality=95)
            self._update_status(f"Gespeichert: {path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")

    # ── DCP Profile Building ──────────────────────────────────

    def _build_dcp_profile(self) -> Optional[DCPProfile]:
        mix = self._get_mix_matrix()

        camera_model = self.manual_camera_var.get().strip()
        selected_camera: Optional[CameraInfo] = None

        if not camera_model:
            cam_name = self.camera_var.get()
            if cam_name and cam_name in self.camera_map:
                selected_camera = self.camera_map[cam_name]
                camera_model = selected_camera.unique_camera_model

        if not camera_model:
            messagebox.showwarning(
                "Kamera fehlt",
                "Bitte wähle eine Kamera aus der Liste oder gib\n"
                "den Kamera-Modellnamen manuell ein.\n\n"
                "Der Name muss exakt dem EXIF-Modellnamen entsprechen,\n"
                "z.B. 'Canon EOS 5D Mark IV' oder 'ILCE-7M3'.")
            return None

        prof_name = self.profile_name_var.get() or f"Channel Swap {mix.name}"

        if self.loaded_dcp is not None:
            profile = DCPProfile(
                camera_model=camera_model,
                profile_name=prof_name,
                color_matrix_1=apply_to_color_matrix(self.loaded_dcp.color_matrix_1, mix),
                color_matrix_2=apply_to_color_matrix(self.loaded_dcp.color_matrix_2, mix)
                    if self.loaded_dcp.color_matrix_2 is not None else None,
                forward_matrix_1=apply_to_forward_matrix(self.loaded_dcp.forward_matrix_1, mix)
                    if self.loaded_dcp.forward_matrix_1 is not None else None,
                forward_matrix_2=apply_to_forward_matrix(self.loaded_dcp.forward_matrix_2, mix)
                    if self.loaded_dcp.forward_matrix_2 is not None else None,
                illuminant_1=self.loaded_dcp.illuminant_1,
                illuminant_2=self.loaded_dcp.illuminant_2,
                tone_curve_data=self.loaded_dcp.tone_curve_data,
                tone_curve_count=self.loaded_dcp.tone_curve_count,
                look_table_dims=self.loaded_dcp.look_table_dims,
                look_table_data=self.loaded_dcp.look_table_data,
            )
            # HueSatMap remapping (#13)
            if self.loaded_dcp.hue_sat_map_dims and self.loaded_dcp.hue_sat_map_data_1:
                try:
                    profile.hue_sat_map_dims = self.loaded_dcp.hue_sat_map_dims
                    profile.hue_sat_map_data_1 = remap_hue_sat_map(
                        self.loaded_dcp.hue_sat_map_data_1,
                        self.loaded_dcp.hue_sat_map_dims, mix)
                    if self.loaded_dcp.hue_sat_map_data_2:
                        profile.hue_sat_map_data_2 = remap_hue_sat_map(
                            self.loaded_dcp.hue_sat_map_data_2,
                            self.loaded_dcp.hue_sat_map_dims, mix)
                except Exception:
                    # Fallback: pass through unchanged
                    profile.hue_sat_map_dims = self.loaded_dcp.hue_sat_map_dims
                    profile.hue_sat_map_data_1 = self.loaded_dcp.hue_sat_map_data_1
                    profile.hue_sat_map_data_2 = self.loaded_dcp.hue_sat_map_data_2

        elif selected_camera is not None:
            cm1 = selected_camera.color_matrix_a
            cm2 = selected_camera.color_matrix_d65

            if cm1 is None and cm2 is None:
                messagebox.showerror("Fehler", "Keine Farbmatrizen für diese Kamera verfügbar.")
                return None

            profile = DCPProfile(
                camera_model=camera_model, profile_name=prof_name,
                illuminant_1=ILLUMINANT_A, illuminant_2=ILLUMINANT_D65)

            if cm1 is not None:
                profile.color_matrix_1 = apply_to_color_matrix(cm1, mix)
            if cm2 is not None:
                profile.color_matrix_2 = apply_to_color_matrix(cm2, mix)
                if cm1 is None:
                    profile.color_matrix_1 = profile.color_matrix_2
                    profile.color_matrix_2 = None
                    profile.illuminant_1 = ILLUMINANT_D65
        else:
            identity = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ])
            cm = np.linalg.inv(identity)
            profile = DCPProfile(
                camera_model=camera_model, profile_name=prof_name,
                color_matrix_1=apply_to_color_matrix(cm, mix),
                illuminant_1=ILLUMINANT_D65)

        profile.copyright = "DNG Channel Tool"
        return profile

    def _export_dcp(self):
        profile = self._build_dcp_profile()
        if not profile:
            return

        mix = self._get_mix_matrix()
        default_name = f"{profile.camera_model.replace(' ', '_')}_{mix.name}.dcp"

        path = filedialog.asksaveasfilename(
            title="DCP-Profil exportieren", defaultextension=".dcp",
            initialfile=default_name,
            filetypes=[("DNG Camera Profile", "*.dcp")])
        if not path:
            return

        try:
            DCPWriter().write(path, profile)
            self._update_status(f"DCP exportiert: {path}")
            messagebox.showinfo("Erfolg",
                f"DCP-Profil gespeichert:\n{path}\n\n"
                f"Kamera: {profile.camera_model}\n"
                f"Profil: {profile.profile_name}\n"
                f"Kanal-Mapping: {mix.name}")
        except Exception as e:
            messagebox.showerror("Fehler", f"DCP-Export fehlgeschlagen:\n{e}")

    def _export_xmp(self):
        """Export XMP preset for Lightroom (Feature #14)."""
        if not HAS_XMP:
            return

        profile = self._build_dcp_profile()
        if not profile:
            return

        mix = self._get_mix_matrix()
        default_name = f"{profile.profile_name.replace(' ', '_')}.xmp"

        path = filedialog.asksaveasfilename(
            title="XMP-Preset exportieren", defaultextension=".xmp",
            initialfile=default_name,
            filetypes=[("XMP Preset", "*.xmp")])
        if not path:
            return

        try:
            write_xmp_preset(path, profile.profile_name, profile.camera_model)
            self._update_status(f"XMP exportiert: {path}")
            messagebox.showinfo("Erfolg",
                f"XMP-Preset gespeichert:\n{path}\n\n"
                f"Das Preset verweist auf DCP-Profil: {profile.profile_name}\n"
                f"Stelle sicher, dass das DCP-Profil auch installiert ist.")
        except Exception as e:
            messagebox.showerror("Fehler", f"XMP-Export fehlgeschlagen:\n{e}")

    def _install_to_adobe(self):
        profile = self._build_dcp_profile()
        if not profile:
            return

        mix = self._get_mix_matrix()
        filename = f"{profile.camera_model.replace(' ', '_')}_{mix.name}.dcp"

        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), filename)

        try:
            DCPWriter().write(tmp_path, profile)
            dest = install_dcp_to_adobe(tmp_path)
            os.unlink(tmp_path)

            self._update_status(f"DCP installiert: {dest}")
            messagebox.showinfo("Erfolg",
                f"DCP-Profil installiert!\n\n"
                f"Pfad: {dest}\n\n"
                f"Starte Lightroom / Camera Raw neu,\n"
                f"dann findest du das Profil unter:\n"
                f"Entwickeln → Profil → Durchsuchen → Channel Swap")
        except Exception as e:
            messagebox.showerror("Fehler", f"Installation fehlgeschlagen:\n{e}")

    # ── Camera Selection ──

    def _on_camera_selected(self, event=None):
        cam_name = self.camera_var.get()
        if cam_name in self.camera_map:
            cam = self.camera_map[cam_name]
            self.manual_camera_var.set(cam.unique_camera_model)
            self.loaded_dcp = None
            mix = self._get_mix_matrix()
            self.profile_name_var.set(f"{cam.clean_make} {cam.clean_model} {mix.name}")
            self._update_matrix_display()
            self._update_status(f"Kamera: {cam.display_name}")

    # ── Undo / Redo ────────────────────────────────────────────

    def _push_undo(self, description: str = ""):
        """Speichert aktuellen Zustand für Undo."""
        if self.preview_image is not None:
            mix = self._get_mix_matrix()
            self.undo_manager.push(self.preview_image, mix.matrix, description or mix.name)

    def _undo(self):
        """Stellt den vorherigen Zustand wieder her."""
        state = self.undo_manager.undo()
        if state is not None:
            self.preview_image = state.preview_image.copy()
            # Mix-Matrix in UI wiederherstellen
            self.mix_mode.set(True)
            for i in range(3):
                for j in range(3):
                    self.mix_vars[i][j].set(float(state.mix_matrix[i, j]))
            self._display_preview()
            self._update_histogram()
            self._update_matrix_display()
            self._update_status(f"Rückgängig: {state.description}")
            logger.debug("Undo: %s", state.description)

    def _redo(self):
        """Stellt den nächsten Zustand wieder her."""
        state = self.undo_manager.redo()
        if state is not None:
            self.preview_image = state.preview_image.copy()
            self.mix_mode.set(True)
            for i in range(3):
                for j in range(3):
                    self.mix_vars[i][j].set(float(state.mix_matrix[i, j]))
            self._display_preview()
            self._update_histogram()
            self._update_matrix_display()
            self._update_status(f"Wiederhergestellt: {state.description}")
            logger.debug("Redo: %s", state.description)

    def _update_undo_menu_state(self):
        """Aktualisiert die Undo/Redo-Menüeinträge."""
        if hasattr(self, '_edit_menu'):
            undo_label = (f"Rückgängig: {self.undo_manager.undo_description}"
                          if self.undo_manager.can_undo else "Rückgängig")
            redo_label = (f"Wiederherstellen: {self.undo_manager.redo_description}"
                          if self.undo_manager.can_redo else "Wiederherstellen")
            self._edit_menu.entryconfig(0,
                state='normal' if self.undo_manager.can_undo else 'disabled',
                label=undo_label)
            self._edit_menu.entryconfig(1,
                state='normal' if self.undo_manager.can_redo else 'disabled',
                label=redo_label)

    # ── Preview / Zoom / Pan ──────────────────────────────────

    def _display_preview(self):
        if self.preview_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        ih, iw = self.preview_image.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self.zoom_level

        # Zoomed image size
        disp_w = int(iw * scale)
        disp_h = int(ih * scale)

        if self.split_view.get() and self.original_image is not None:
            # Split view (#5): left = original, right = swapped
            self._display_split_preview(cw, ch, scale, disp_w, disp_h)
        else:
            self._display_single_preview(cw, ch, scale, disp_w, disp_h)

        # Update zoom display
        pct = int(scale / base_scale * 100) if base_scale > 0 else 100
        self.zoom_var.set(f"{pct}%")

    def _display_single_preview(self, cw, ch, scale, disp_w, disp_h):
        ih, iw = self.preview_image.shape[:2]

        img = Image.fromarray(self.preview_image)
        img = img.resize((max(1, disp_w), max(1, disp_h)), Image.LANCZOS)

        # Apply pan offset
        ox = int(cw / 2 - disp_w / 2 + self.pan_x)
        oy = int(ch / 2 - disp_h / 2 + self.pan_y)

        self.photo_image = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, image=self.photo_image, anchor=tk.NW)

    def _display_split_preview(self, cw, ch, scale, disp_w, disp_h):
        """Draw split view: left=original, right=swapped (#5)."""
        half = cw // 2

        # Original
        orig_img = Image.fromarray(self.original_image)
        orig_img = orig_img.resize((max(1, disp_w), max(1, disp_h)), Image.LANCZOS)

        # Swapped
        swap_img = Image.fromarray(self.preview_image)
        swap_img = swap_img.resize((max(1, disp_w), max(1, disp_h)), Image.LANCZOS)

        # Combine into one image
        combined = Image.new('RGB', (cw, ch), '#2b2b2b')

        ox = int(half - disp_w / 2 + self.pan_x)
        oy = int(ch / 2 - disp_h / 2 + self.pan_y)

        # Paste original on left half
        combined.paste(orig_img, (ox - half, oy))
        # Paste swapped on right half
        combined.paste(swap_img, (ox, oy))

        self.photo_image = ImageTk.PhotoImage(combined)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)

        # Divider line
        self.canvas.create_line(half, 0, half, ch, fill='#ffffff', width=2)
        self.canvas.create_text(half - 40, 15, text="Vorher", fill='#aaaaaa',
                                font=('Arial', 10))
        self.canvas.create_text(half + 40, 15, text="Nachher", fill='#aaaaaa',
                                font=('Arial', 10))

    def _on_canvas_resize(self, event):
        if self.preview_image is not None:
            self._display_preview()

    def _on_mouse_wheel(self, event):
        """Zoom with mouse wheel (#7)."""
        if self.original_image is None:
            return

        factor = 1.15 if event.delta > 0 else (1 / 1.15)
        new_zoom = self.zoom_level * factor
        new_zoom = max(0.1, min(20.0, new_zoom))

        # Zoom towards cursor position
        cx = event.x - self.canvas.winfo_width() / 2
        cy = event.y - self.canvas.winfo_height() / 2
        self.pan_x = cx - factor * (cx - self.pan_x)
        self.pan_y = cy - factor * (cy - self.pan_y)

        self.zoom_level = new_zoom
        self._display_preview()

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _on_pan_move(self, event):
        if self._drag_start is None:
            return
        sx, sy, spx, spy = self._drag_start
        self.pan_x = spx + (event.x - sx)
        self.pan_y = spy + (event.y - sy)
        self._display_preview()

    def _on_pan_end(self, event):
        self._drag_start = None

    def _reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if self.preview_image is not None:
            self._display_preview()

    # ── Histogram (#6) ────────────────────────────────────────

    def _update_histogram(self):
        if self.show_histogram.get():
            self.histogram.update_histogram(self.preview_image)

    def _on_histogram_toggle(self):
        if self.show_histogram.get():
            self.histogram.pack(fill=tk.X)
            self._update_histogram()
        else:
            self.histogram.pack_forget()

    # ── View ──

    def _toggle_split(self):
        self.split_view.set(not self.split_view.get())
        self._on_view_changed()

    def _on_view_changed(self):
        if self.preview_image is not None:
            self._display_preview()

    # ── Matrix Display ────────────────────────────────────────

    def _update_matrix_display(self):
        mix = self._get_mix_matrix()
        m = mix.matrix
        lines = [f"Aktive Matrix ({mix.name}):"]
        labels = ['R', 'G', 'B']
        for i in range(3):
            vals = '  '.join(f'{m[i,j]:+.2f}' for j in range(3))
            lines.append(f"  {labels[i]}_aus = [{vals}]")

        cam_name = self.camera_var.get()
        cam = self.camera_map.get(cam_name)
        if cam and cam.color_matrix_d65 is not None:
            lines.append("")
            lines.append("ColorMatrix D65 (Original):")
            for row in cam.color_matrix_d65:
                lines.append(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")
            swapped = apply_to_color_matrix(cam.color_matrix_d65, mix)
            if swapped is not None:
                lines.append(f"ColorMatrix D65 ({mix.name}):")
                for row in swapped:
                    lines.append(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")

        elif self.loaded_dcp and self.loaded_dcp.color_matrix_1 is not None:
            lines.append("")
            lines.append("DCP ColorMatrix 1 (Original):")
            for row in self.loaded_dcp.color_matrix_1:
                lines.append(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")
            swapped = apply_to_color_matrix(self.loaded_dcp.color_matrix_1, mix)
            if swapped is not None:
                lines.append(f"ColorMatrix 1 ({mix.name}):")
                for row in swapped:
                    lines.append(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")

        self.matrix_label.config(text="\n".join(lines))

    def _update_status(self, text: str):
        self.status_var.set(text)

    # ── Drag & Drop (Fix #1, #2) ─────────────────────────────

    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)
        except ImportError:
            pass

    @staticmethod
    def _parse_drop_paths(data: str) -> list:
        """Parst TkDND Drop-Daten in eine Liste von Dateipfaden."""
        paths = []
        i = 0
        while i < len(data):
            if data[i] == '{':
                end = data.find('}', i)
                if end == -1:
                    break
                paths.append(data[i + 1:end])
                i = end + 2
            elif data[i] == ' ':
                i += 1
            else:
                end = data.find(' ', i)
                if end == -1:
                    end = len(data)
                paths.append(data[i:end])
                i = end + 1
        return paths

    def _on_drop(self, event):
        paths = self._parse_drop_paths(event.data)
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.dcp':
                self._open_dcp(path)
                return
            elif ext in ('.npc', '.np3', '.ncp') and HAS_NPC:
                try:
                    from npc_io import read_npc
                    pc = read_npc(path)
                    NikonPresetCreatorDialog(self.root, self, pc)
                except Exception:
                    pass
                return
            elif ext in ('.nef', '.nrw') and HAS_NEF_EXTRACT:
                self._load_image_threaded(path)
                return
            elif ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff',
                         '.bmp', '.dng', '.cr2', '.cr3', '.arw',
                         '.orf', '.raf', '.rw2', '.pef', '.nef', '.nrw'):
                self._load_image_threaded(path)
                return

    # ── NEF Picture Control Extractor ────────────────────────

    def _extract_nef_picture_control(self):
        """Öffnet NEF-Datei und zeigt extrahierten Picture Control im Dialog."""
        if not HAS_NEF_EXTRACT:
            messagebox.showinfo("Info", "NEF-Extraktor nicht verfügbar.\npip install exifread")
            return

        path = filedialog.askopenfilename(
            title="NEF-Datei mit Picture Control öffnen",
            filetypes=[
                ("Nikon RAW", "*.nef *.nrw"),
                ("Alle Dateien", "*.*"),
            ])
        if not path:
            return

        self._update_status(f"Extrahiere Picture Control: {os.path.basename(path)}…")
        self.root.update()

        try:
            pc = extract_picture_control(path)
            NEFExtractDialog(self.root, self, pc, path)
        except Exception as e:
            messagebox.showerror("Fehler",
                f"Picture Control konnte nicht extrahiert werden:\n{e}")
        self._update_status("Bereit.")

    def _quick_nef_to_preset(self):
        """NEF öffnen → sofort als Lightroom-Preset speichern."""
        if not HAS_NEF_EXTRACT:
            return

        path = filedialog.askopenfilename(
            title="NEF-Datei wählen",
            filetypes=[("Nikon RAW", "*.nef *.nrw"), ("Alle Dateien", "*.*")])
        if not path:
            return

        try:
            pc = extract_picture_control(path)
            preset_name = pc.name or "Nikon Preset"

            # Direkt in Adobe installieren
            preset_dir = os.path.join(
                os.environ.get('APPDATA', ''),
                'Adobe', 'CameraRaw', 'Settings', 'Nikon Picture Controls')
            os.makedirs(preset_dir, exist_ok=True)

            xmp_path = os.path.join(preset_dir, f"{preset_name}.xmp")
            picture_control_to_xmp(pc, xmp_path, preset_name)

            self._update_status(f"Preset installiert: {preset_name}")
            messagebox.showinfo("Erfolg",
                f"Picture Control \"{preset_name}\" extrahiert!\n\n"
                f"Basis: {pc.base}\n"
                f"Tonkurve: {len(pc.tone_curve)} Punkte\n"
                f"Monochrom: {'Ja' if pc.is_monochrome else 'Nein'}\n\n"
                f"Installiert als Lightroom-Preset:\n{xmp_path}\n\n"
                f"Starte Lightroom neu, dann findest du es unter:\n"
                f"Vorgaben → Nikon Picture Controls → {preset_name}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Extraktion fehlgeschlagen:\n{e}")

    # ── Style Transfer ────────────────────────────────────────

    def _analyze_style(self):
        """Analysiert den Bildstil des aktuell geladenen Bildes."""
        if not HAS_STYLE:
            return
        if self.original_image is None:
            # Bild laden falls keins da ist
            path = filedialog.askopenfilename(
                title="Referenzbild für Stil-Analyse öffnen",
                filetypes=[("Alle Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
            if not path:
                return
            img = np.array(Image.open(path).convert('RGB'))
        else:
            img = self.original_image
            path = self.current_file or "Aktuelles Bild"

        self._update_status("Analysiere Bildstil…")
        self.root.update()

        style = analyze_image(img, os.path.basename(path).split('.')[0])
        StyleResultDialog(self.root, self, style)
        self._update_status("Stil-Analyse abgeschlossen.")

    def _transfer_style(self):
        """Überträgt den Stil eines Referenzbildes auf das aktuelle Bild."""
        if not HAS_STYLE or self.original_image is None:
            messagebox.showinfo("Info",
                "Bitte zuerst ein Zielbild laden (Datei → Bild öffnen).")
            return

        path = filedialog.askopenfilename(
            title="Referenzbild (Stilvorlage) öffnen",
            filetypes=[("Alle Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if not path:
            return

        self._update_status(f"Übertrage Stil von {os.path.basename(path)}…")
        self.root.update()

        ref_img = np.array(Image.open(path).convert('RGB'))
        style = analyze_image(ref_img, os.path.basename(path).split('.')[0])

        # Stil auf aktuelles Bild anwenden
        self._push_undo("Stil-Transfer")
        self.preview_image = apply_style(self.original_image, style)
        self._display_preview()
        self._update_histogram()

        StyleResultDialog(self.root, self, style)
        self._update_status(f"Stil übertragen: {style.name}")

    def _compare_style(self):
        """Vergleicht Original + bearbeitetes Bild → extrahiert Transformation als Preset."""
        if not HAS_STYLE:
            return

        messagebox.showinfo("Anleitung",
            "Bitte wähle nacheinander:\n"
            "1. Das ORIGINAL-Bild (unbearbeitet)\n"
            "2. Das BEARBEITETE Bild (mit dem gewünschten Stil)\n\n"
            "Die Differenz wird als Preset extrahiert.")

        orig_path = filedialog.askopenfilename(
            title="1. Original-Bild (unbearbeitet) öffnen",
            filetypes=[("Alle Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if not orig_path:
            return

        styled_path = filedialog.askopenfilename(
            title="2. Bearbeitetes Bild (mit Stil) öffnen",
            filetypes=[("Alle Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if not styled_path:
            return

        self._update_status("Vergleiche Bilder und berechne Transformation…")
        self.root.update()

        orig = np.array(Image.open(orig_path).convert('RGB'))
        styled = np.array(Image.open(styled_path).convert('RGB'))

        name = os.path.basename(styled_path).split('.')[0]
        style = compare_images(orig, styled, name)

        # Zeige Vorher/Nachher
        self.original_image = orig
        self.preview_image = styled
        self.current_file = orig_path
        self.split_view.set(True)
        self._display_preview()
        self._update_histogram()

        StyleResultDialog(self.root, self, style)
        self._update_status(f"Stil-Vergleich: {style.name}")

    # ── Fujifilm Rezepte ──────────────────────────────────────

    def _open_fuji_recipe_dialog(self):
        """Öffnet den Fujifilm-Rezept-Konverter."""
        if HAS_FUJI:
            FujiRecipeDialog(self.root, self)

    def _create_canon_sony_preset(self):
        """Canon/Sony Preset erstellen (#3)."""
        if not HAS_CANON_SONY:
            return
        messagebox.showinfo("Canon/Sony Presets",
            "Canon Picture Styles und Sony Creative Looks\n"
            "können als Lightroom XMP-Presets exportiert werden.\n\n"
            "Wähle einen Basisstil und passe die Parameter an.")
        # TODO: Full dialog like NikonPresetCreatorDialog

    # ── 3D LUT Export (#1) ──

    def _export_lut(self):
        """Exportiert aktuelle Einstellungen als 3D LUT (.cube)."""
        if not HAS_LUT:
            return

        mix = self._get_mix_matrix()
        path = filedialog.asksaveasfilename(
            title="3D LUT exportieren",
            defaultextension=".cube",
            initialfile=f"ChannelTool_{mix.name}.cube",
            filetypes=[("3D LUT", "*.cube")])
        if not path:
            return

        self._update_status("Generiere 3D LUT…")
        self.root.update()

        try:
            lut = mix_matrix_to_lut(mix.matrix, size=33, title=mix.name)
            write_cube_lut(path, lut, title=f"DNG Channel Tool - {mix.name}")
            self._update_status(f"LUT exportiert: {path}")
            messagebox.showinfo("Erfolg",
                f"3D LUT gespeichert:\n{path}\n\n"
                f"Verwendbar in: DaVinci Resolve, Premiere Pro,\n"
                f"Photoshop, Capture One, OBS, etc.")
        except Exception as e:
            messagebox.showerror("Fehler", f"LUT-Export fehlgeschlagen:\n{e}")

    # ── ICC Profile Export (#7) ──

    def _export_icc(self):
        """Exportiert als ICC-Profil."""
        if not HAS_ICC:
            return

        mix = self._get_mix_matrix()
        path = filedialog.asksaveasfilename(
            title="ICC-Profil exportieren",
            defaultextension=".icc",
            initialfile=f"ChannelTool_{mix.name}.icc",
            filetypes=[("ICC Profile", "*.icc *.icm")])
        if not path:
            return

        try:
            channel_swap_to_icc(path, mix.matrix, f"DNG Channel Tool {mix.name}")
            self._update_status(f"ICC-Profil exportiert: {path}")
            messagebox.showinfo("Erfolg", f"ICC-Profil gespeichert:\n{path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"ICC-Export fehlgeschlagen:\n{e}")

    # ── WB-Pipette (#2) ──

    def _toggle_wb_picker(self):
        """Aktiviert/deaktiviert die Weißabgleich-Pipette."""
        if not HAS_WB:
            return
        self.wb_picker_active = not self.wb_picker_active
        if self.wb_picker_active:
            self.canvas.config(cursor='crosshair')
            self._update_status("WB-Pipette aktiv – Klicke auf einen neutralgrauen Punkt. [W] zum Beenden.")
            self.canvas.bind('<ButtonPress-1>', self._on_wb_pick)
        else:
            self.canvas.config(cursor='')
            self._update_status("WB-Pipette deaktiviert.")
            self.canvas.bind('<ButtonPress-1>', self._on_pan_start)

    def _on_wb_pick(self, event):
        """WB-Pipette: Klick auf Bild → WB berechnen."""
        if not self.wb_picker_active or self.original_image is None:
            return

        # Canvas-Koordinaten → Bild-Koordinaten
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        ih, iw = self.original_image.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self.zoom_level

        ox = cw / 2 - iw * scale / 2 + self.pan_x
        oy = ch / 2 - ih * scale / 2 + self.pan_y

        img_x = int((event.x - ox) / scale)
        img_y = int((event.y - oy) / scale)

        if 0 <= img_x < iw and 0 <= img_y < ih:
            wb = calculate_wb_from_pixel(self.original_image, img_x, img_y)

            self._push_undo("Weißabgleich")
            # WB-Korrektur anwenden
            corrected = apply_wb_correction(
                self.original_image, wb['r_gain'], wb['g_gain'], wb['b_gain'])
            self.preview_image = corrected
            self._display_preview()
            self._update_histogram()

            self._toggle_wb_picker()  # Deaktivieren

            messagebox.showinfo("Weißabgleich",
                f"Gemessene Farbe: RGB({wb['measured_rgb'][0]}, "
                f"{wb['measured_rgb'][1]}, {wb['measured_rgb'][2]})\n\n"
                f"Korrektur:\n"
                f"  R-Gain: {wb['r_gain']:.3f}\n"
                f"  B-Gain: {wb['b_gain']:.3f}\n"
                f"  Temperatur: ~{wb['correction_temp']}K\n"
                f"  Tint: {wb['correction_tint']:+d}\n\n"
                f"Lightroom-Werte:\n"
                f"  Temperature: {wb['correction_temp']}\n"
                f"  Tint: {wb['correction_tint']}")

    # ── Histogram Matching (#6) ──

    # ── Infrarot-Tools ─────────────────────────────────────

    def _ir_wb_picker(self):
        """IR-Weißabgleich: Klick auf Vegetation."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein IR-Bild laden.")
            return

        self.wb_picker_active = True
        self.canvas.config(cursor='crosshair')
        self._update_status(
            "IR-WB: Klicke auf VEGETATION (soll weiß werden). [W] zum Abbrechen.")

        def _on_ir_wb(event):
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            ih, iw = self.original_image.shape[:2]
            base_scale = min(cw / iw, ch / ih)
            scale = base_scale * self.zoom_level
            ox = cw / 2 - iw * scale / 2 + self.pan_x
            oy = ch / 2 - ih * scale / 2 + self.pan_y
            img_x = int((event.x - ox) / scale)
            img_y = int((event.y - oy) / scale)

            if 0 <= img_x < iw and 0 <= img_y < ih:
                wb = calculate_ir_wb(self.original_image, img_x, img_y)
                self._push_undo("IR-Weißabgleich")
                corrected = apply_ir_wb(self.original_image, wb)
                self.preview_image = corrected
                self._display_preview()
                self._update_histogram()

                self.canvas.config(cursor='')
                self.wb_picker_active = False
                self.canvas.bind('<ButtonPress-1>', self._on_pan_start)

                messagebox.showinfo("IR-Weißabgleich",
                    f"Vegetation gemessen: RGB({wb['measured_rgb'][0]}, "
                    f"{wb['measured_rgb'][1]}, {wb['measured_rgb'][2]})\n\n"
                    f"Korrektur:\n"
                    f"  R: ×{wb['r_gain']:.2f}  G: ×{wb['g_gain']:.2f}  B: ×{wb['b_gain']:.2f}\n\n"
                    f"Lightroom-Werte:\n"
                    f"  Temperature: {wb['lr_temperature']}K\n"
                    f"  Tint: {wb['lr_tint']}")

        self.canvas.bind('<ButtonPress-1>', _on_ir_wb)

    def _apply_ir_false_color(self, preset_name: str):
        """Wendet ein IR False-Color Preset an."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein IR-Bild laden.")
            return

        preset = IR_FALSE_COLOR_PRESETS.get(preset_name)
        if not preset:
            return

        self._push_undo(f"IR: {preset_name}")
        self.preview_image = apply_ir_preset(self.original_image, preset)
        self._display_preview()
        self._update_histogram()
        self._update_status(f"IR-Preset: {preset_name} ({preset.description[:60]})")

    def _simulate_ir_filter(self, filter_type: str):
        """Simuliert einen IR-Filter auf dem aktuellen Bild."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein Bild laden.")
            return

        self._push_undo(f"IR-Filter: {filter_type}")
        self.preview_image = simulate_ir_filter(self.original_image, filter_type)
        self._display_preview()
        self._update_histogram()
        f = IR_FILTERS[filter_type]
        self._update_status(f"IR-Filter Simulation: {f.name}")

    def _detect_hotspot(self):
        """Erkennt IR-Hotspots im aktuellen Bild."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein IR-Bild laden.")
            return

        result = detect_hotspot(self.original_image)

        if result.has_hotspot:
            fix = messagebox.askyesno("Hotspot erkannt",
                f"{result.description}\n\nHotspot korrigieren?")
            if fix:
                self._push_undo("Hotspot-Korrektur")
                self.preview_image = correct_hotspot(self.original_image, result)
                self._display_preview()
                self._update_histogram()
                self._update_status("Hotspot korrigiert.")
        else:
            messagebox.showinfo("Hotspot-Analyse", result.description)

    def _calculate_ndvi(self):
        """Berechnet NDVI (Vegetationsindex) aus dem IR-Bild."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein IR-Bild laden.")
            return

        ndvi_img = calculate_ndvi(self.original_image)
        stats = ndvi_statistics(self.original_image)

        self.preview_image = ndvi_img
        self._display_preview()
        self._update_histogram()

        messagebox.showinfo("NDVI-Analyse",
            f"Vegetationsindex (NDVI):\n\n"
            f"Min: {stats['ndvi_min']:.2f}  Max: {stats['ndvi_max']:.2f}\n"
            f"Mittel: {stats['ndvi_mean']:.2f}  Std: {stats['ndvi_std']:.2f}\n\n"
            f"Vegetationsbedeckung: {stats['vegetation_coverage']:.1f}%\n"
            f"Gesunde Vegetation: {stats['healthy_vegetation']:.1f}%\n\n"
            f"Farbskala: Rot=kein Grün, Gelb=wenig, Grün=gesund")

    def _generate_ir_dcp(self):
        """Generiert ein DCP-Profil für IR-Fotografie."""
        if not HAS_IR:
            return

        cam = self.manual_camera_var.get() or "Unknown Camera"

        # Filter-Auswahl
        filter_names = list(IR_FILTERS.keys())
        # Simple dialog
        filter_type = "720nm"  # Default

        swap_options = ["RGB (kein Tausch)", "BGR (R↔B)", "BRG", "GBR"]
        swap = "BGR"

        profile = generate_ir_dcp(cam, filter_type, swap)

        path = filedialog.asksaveasfilename(
            title="IR-DCP-Profil speichern",
            defaultextension=".dcp",
            initialfile=f"{cam}_IR_{filter_type}_{swap}.dcp",
            filetypes=[("DNG Camera Profile", "*.dcp")])
        if path:
            try:
                from dcp_io import DCPWriter
                DCPWriter().write(path, profile)
                self._update_status(f"IR-DCP gespeichert: {path}")
                messagebox.showinfo("Erfolg",
                    f"IR-DCP-Profil gespeichert:\n{path}\n\n"
                    f"Kamera: {cam}\nFilter: {filter_type}\nKanaltausch: {swap}")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def _export_ir_preset(self):
        """Exportiert aktuelles IR-Preset als XMP oder LUT."""
        if not HAS_IR:
            return

        # Zeige Preset-Auswahl
        presets = list(IR_FALSE_COLOR_PRESETS.keys())
        if not presets:
            return

        # Alle Presets auf einmal exportieren
        folder = filedialog.askdirectory(title="Ausgabeordner für IR-Presets")
        if not folder:
            return

        count = 0
        for name, preset in IR_FALSE_COLOR_PRESETS.items():
            safe = name.replace(' ', '_').replace('/', '-')
            xmp_path = os.path.join(folder, f"IR_{safe}.xmp")
            ir_preset_to_xmp(preset, xmp_path)
            count += 1

            if HAS_LUT:
                from lut_export import write_cube_lut, combined_lut
                lut = combined_lut(
                    mix_matrix=preset.mix_matrix,
                    tone_curve=preset.tone_curve if preset.tone_curve else None,
                    saturation=preset.saturation,
                    monochrome=preset.monochrome,
                    size=33)
                cube_path = os.path.join(folder, f"IR_{safe}.cube")
                write_cube_lut(cube_path, lut, f"IR {name}")
                count += 1

        messagebox.showinfo("Erfolg",
            f"{count} IR-Preset-Dateien exportiert nach:\n{folder}\n\n"
            f"XMP-Presets für Lightroom + .cube LUTs für Resolve/Premiere")

    def _histogram_match(self):
        """Passt die Farbverteilung an ein Referenzbild an."""
        if not HAS_WB or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein Bild laden.")
            return

        path = filedialog.askopenfilename(
            title="Referenzbild für Histogram Matching",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if not path:
            return

        ref = np.array(Image.open(path).convert('RGB'))
        self._push_undo("Histogram Matching")
        self.preview_image = histogram_match(self.original_image, ref)
        self._display_preview()
        self._update_histogram()
        self._update_status(f"Histogram Matching: {os.path.basename(path)}")

    # ── Kamera-JPEG vs RAW (#8) ──

    def _compare_jpeg_raw(self):
        """Vergleicht eingebettetes Kamera-JPEG mit RAW-Entwicklung."""
        if not HAS_WB:
            return

        path = filedialog.askopenfilename(
            title="RAW-Datei für JPEG-Vergleich",
            filetypes=[("RAW", "*.nef *.cr2 *.cr3 *.arw *.dng *.orf *.rw2 *.raf")])
        if not path:
            return

        self._update_status("Extrahiere Kamera-JPEG und entwickle RAW…")
        self.root.update()

        result = compare_jpeg_vs_raw(path)
        if result is None:
            messagebox.showinfo("Info", "Konnte kein JPEG aus der RAW-Datei extrahieren.")
            return

        camera_jpeg, raw_dev = result
        self.original_image = raw_dev
        self.preview_image = camera_jpeg
        self.current_file = path
        self.split_view.set(True)
        self._display_preview()
        self._update_histogram()
        self._update_status(
            f"Links: RAW-Entwicklung | Rechts: Kamera-JPEG ({os.path.basename(path)})")

    # ── Color Checker (#5) ──

    def _color_checker_calibrate(self):
        """Kalibrierung mit Color Checker Foto."""
        if not HAS_CHECKER:
            return

        path = filedialog.askopenfilename(
            title="Foto der Farbkarte (ColorChecker) öffnen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if not path:
            return

        img = np.array(Image.open(path).convert('RGB'))
        self._update_status("Kalibriere mit ColorChecker…")
        self.root.update()

        try:
            result = calibrate_from_colorchecker(img)

            info = (
                f"Color Checker Kalibrierung abgeschlossen!\n\n"
                f"Gemessene Felder: {len(result.patches)}\n"
                f"Durchschnittl. Farbabstand: {result.avg_delta_e:.1f}\n"
                f"Maximaler Farbabstand: {result.max_delta_e:.1f}\n\n"
                f"Korrekturmatrix:\n"
            )
            M = result.correction_matrix
            for row in M:
                info += f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]\n"

            save = messagebox.askyesno("Kalibrierung", info + "\nAls DCP-Profil speichern?")
            if save:
                cam = self.manual_camera_var.get() or "Unknown Camera"
                dcp_path = filedialog.asksaveasfilename(
                    title="DCP-Profil speichern",
                    defaultextension=".dcp",
                    initialfile=f"{cam}_calibrated.dcp",
                    filetypes=[("DNG Camera Profile", "*.dcp")])
                if dcp_path:
                    from dcp_io import DCPWriter
                    profile = calibration_to_dcp(result, cam, "ColorChecker Kalibriert")
                    DCPWriter().write(dcp_path, profile)
                    self._update_status(f"Kalibriertes DCP gespeichert: {dcp_path}")

        except Exception as e:
            messagebox.showerror("Fehler", f"Kalibrierung fehlgeschlagen:\n{e}")

    # ── Preset-Bibliothek (#4) ──

    def _open_preset_library(self):
        """Öffnet die Preset-Bibliothek."""
        if not HAS_LIBRARY:
            return
        PresetLibraryDialog(self.root, self)

    def _show_camera_info(self):
        """Zeigt Kamera-Info des aktuell geladenen Bildes."""
        if not self.current_file:
            messagebox.showinfo("Info", "Kein Bild geladen.")
            return

        info_lines = [f"Datei: {os.path.basename(self.current_file)}"]
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            info_lines.append(f"Größe: {w} × {h}")
        if self.detected_camera:
            info_lines.append(f"Kamera: {self.detected_camera}")

        # Try to get more EXIF info
        try:
            import exifread
            with open(self.current_file, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            for key in ['Image Make', 'Image Model', 'EXIF LensModel',
                        'EXIF FocalLength', 'EXIF FNumber', 'EXIF ExposureTime',
                        'EXIF ISOSpeedRatings', 'MakerNote Whitebalance',
                        'MakerNote ColorSpace', 'MakerNote ActiveDLighting']:
                if key in tags:
                    label = key.split(' ', 1)[-1]
                    info_lines.append(f"{label}: {tags[key]}")

            pc_tag = tags.get('MakerNote PictureControl')
            if pc_tag:
                vals = pc_tag.values
                name = bytes(vals[8:28]).decode('ascii', errors='replace').rstrip('\x00').strip()
                base = bytes(vals[28:48]).decode('ascii', errors='replace').rstrip('\x00').strip()
                info_lines.append(f"Picture Control: {name} ({base})")
        except Exception:
            pass

        messagebox.showinfo("Kamera-Info", "\n".join(info_lines))

    # ── Nikon Preset Creator ─────────────────────────────────

    def _create_nikon_preset(self):
        """Öffnet den Nikon Picture Control Creator Dialog."""
        if not HAS_NPC:
            return
        NikonPresetCreatorDialog(self.root, self)

    def _open_npc_file(self):
        """Lädt eine bestehende NPC/NP3-Datei und zeigt sie im Creator."""
        if not HAS_NPC:
            return
        path = filedialog.askopenfilename(
            title="Nikon Preset laden",
            filetypes=[
                ("Nikon Picture Control", "*.npc *.NPC *.np3 *.NP3 *.ncp *.NCP"),
                ("Alle Dateien", "*.*"),
            ])
        if path:
            try:
                pc = read_npc(path)
                NikonPresetCreatorDialog(self.root, self, pc)
                self._update_status(f"NPC geladen: {pc.name}")
            except Exception as e:
                messagebox.showerror("Fehler", f"NPC-Datei konnte nicht geladen werden:\n{e}")

    def _install_npc_to_card(self):
        """Kopiert eine NPC/NP3-Datei auf eine SD-Karte."""
        if not HAS_NPC:
            return

        path = filedialog.askopenfilename(
            title="NPC/NP3-Datei wählen",
            filetypes=[("Nikon Picture Control", "*.npc *.NPC *.np3 *.NP3 *.ncp *.NCP")])
        if not path:
            return

        cards = find_sd_cards()
        if not cards:
            # Manual selection
            card = filedialog.askdirectory(title="SD-Karte / Laufwerk wählen")
            if not card:
                return
            cards = [card]

        try:
            dest = install_to_camera(path, cards[0])
            messagebox.showinfo("Erfolg",
                f"Nikon Preset auf SD-Karte kopiert!\n\n"
                f"Pfad: {dest}\n\n"
                f"In der Kamera:\n"
                f"Aufnahme-Menü → Bildstile verwalten\n"
                f"→ Laden/Speichern → Von Karte laden")
        except Exception as e:
            messagebox.showerror("Fehler", f"Kopieren fehlgeschlagen:\n{e}")

    # ── DCP Kameramodell umschreiben ────────────────────────

    def _rewrite_dcp_model(self):
        """Dialog zum Umschreiben des Kameramodells in DCP-Profilen."""
        src_path = filedialog.askopenfilename(
            title="DCP-Profil wählen",
            filetypes=[("DCP-Profil", "*.dcp"), ("Alle Dateien", "*.*")])
        if not src_path:
            return

        # Aktuelles Kameramodell auslesen
        try:
            reader = DCPReader()
            profile = reader.read(src_path)
        except Exception as e:
            messagebox.showerror("Fehler", f"DCP konnte nicht gelesen werden:\n{e}")
            return

        # Dialog für neues Kameramodell
        dlg = tk.Toplevel(self.root)
        dlg.title("DCP-Kameramodell umschreiben")
        dlg.geometry("500x280")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = ttk.Frame(dlg, padding=15)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text=f"Datei: {os.path.basename(src_path)}",
                  font=('', 9, 'bold')).pack(anchor='w')
        ttk.Label(frame, text=f"Profilname: {profile.profile_name}").pack(anchor='w', pady=(2, 0))
        ttk.Label(frame, text=f"Aktuelles Modell: {profile.camera_model}").pack(anchor='w', pady=(2, 10))

        ttk.Label(frame, text="Neues Kameramodell (exakt wie in EXIF):").pack(anchor='w')
        new_model_var = tk.StringVar(value=profile.camera_model)
        model_entry = ttk.Entry(frame, textvariable=new_model_var, width=50)
        model_entry.pack(anchor='w', pady=(2, 5))
        model_entry.select_range(0, 'end')
        model_entry.focus_set()

        ttk.Label(frame, text="Neuer Profilname (leer = automatisch):").pack(anchor='w')
        new_name_var = tk.StringVar()
        ttk.Entry(frame, textvariable=new_name_var, width=50).pack(anchor='w', pady=(2, 10))

        # Bekannte Nikon Z-Modelle als Schnellauswahl
        quick_frame = ttk.LabelFrame(frame, text="Nikon Z Schnellauswahl", padding=5)
        quick_frame.pack(fill='x', pady=(0, 10))
        nikon_models = [
            "NIKON Z 5", "NIKON Z 6", "NIKON Z 6_2", "NIKON Z 6_3",
            "NIKON Z 7", "NIKON Z 7_2", "NIKON Z 8", "NIKON Z 9",
            "NIKON Z 30", "NIKON Z 50", "NIKON Z 50_2",
            "NIKON Z fc", "NIKON Z f",
        ]
        for i, m in enumerate(nikon_models):
            ttk.Button(quick_frame, text=m.replace("NIKON ", ""),
                       command=lambda model=m: new_model_var.set(model),
                       width=6).grid(row=i // 7, column=i % 7, padx=1, pady=1)

        def do_save():
            new_model = new_model_var.get().strip()
            if not new_model:
                messagebox.showwarning("Fehler", "Kein Kameramodell eingegeben.", parent=dlg)
                return

            new_name = new_name_var.get().strip() or None

            dest_path = filedialog.asksaveasfilename(
                title="DCP speichern als",
                initialfile=os.path.basename(src_path),
                initialdir=os.path.dirname(src_path),
                defaultextension=".dcp",
                filetypes=[("DCP-Profil", "*.dcp")],
                parent=dlg)
            if not dest_path:
                return

            try:
                result = rewrite_dcp_camera_model(src_path, dest_path, new_model, new_name)
                dlg.destroy()
                messagebox.showinfo("Erfolg",
                    f"DCP-Profil umgeschrieben!\n\n"
                    f"Kamera: {profile.camera_model} → {result.camera_model}\n"
                    f"Profil: {result.profile_name}\n"
                    f"Datei: {os.path.basename(dest_path)}")
                self._update_status(f"DCP umgeschrieben: {result.camera_model}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}", parent=dlg)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="Speichern unter…", command=do_save).pack(side='right')
        ttk.Button(btn_frame, text="Abbrechen", command=dlg.destroy).pack(side='right', padx=(0, 5))

    # ── DCP XML Operationen ────────────────────────────────

    def _export_dcp_xml(self):
        """DCP → XML exportieren."""
        if not HAS_DCP_XML:
            return
        src = filedialog.askopenfilename(
            title="DCP-Profil wählen",
            filetypes=[("DCP-Profil", "*.dcp"), ("Alle Dateien", "*.*")])
        if not src:
            return
        base = os.path.splitext(src)[0]
        dest = filedialog.asksaveasfilename(
            title="XML speichern als",
            initialfile=os.path.basename(base) + ".xml",
            initialdir=os.path.dirname(src),
            defaultextension=".xml",
            filetypes=[("XML-Datei", "*.xml")])
        if not dest:
            return
        try:
            profile = export_dcp_to_xml(src, dest)
            messagebox.showinfo("Erfolg",
                f"DCP als XML exportiert!\n\n"
                f"Profil: {profile.profile_name}\n"
                f"Kamera: {profile.camera_model}\n"
                f"Datei: {os.path.basename(dest)}")
            self._update_status(f"XML exportiert: {os.path.basename(dest)}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen:\n{e}")

    def _import_xml_dcp(self):
        """XML → DCP kompilieren."""
        if not HAS_DCP_XML:
            return
        src = filedialog.askopenfilename(
            title="XML-Datei wählen",
            filetypes=[("XML-Datei", "*.xml"), ("Alle Dateien", "*.*")])
        if not src:
            return
        base = os.path.splitext(src)[0]
        dest = filedialog.asksaveasfilename(
            title="DCP speichern als",
            initialfile=os.path.basename(base) + ".dcp",
            initialdir=os.path.dirname(src),
            defaultextension=".dcp",
            filetypes=[("DCP-Profil", "*.dcp")])
        if not dest:
            return
        try:
            profile = import_xml_to_dcp(src, dest)
            messagebox.showinfo("Erfolg",
                f"XML zu DCP kompiliert!\n\n"
                f"Profil: {profile.profile_name}\n"
                f"Kamera: {profile.camera_model}\n"
                f"Datei: {os.path.basename(dest)}")
            self._update_status(f"DCP kompiliert: {os.path.basename(dest)}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Import fehlgeschlagen:\n{e}")

    def _dcp_make_invariant(self):
        """DCP invariant machen (LookTable in HueSatMap mergen)."""
        if not HAS_DCP_XML:
            return
        src = filedialog.askopenfilename(
            title="DCP-Profil wählen",
            filetypes=[("DCP-Profil", "*.dcp"), ("Alle Dateien", "*.*")])
        if not src:
            return
        try:
            profile = DCPReader().read(src)
            if profile.look_table_data is None:
                messagebox.showinfo("Info",
                    "Dieses Profil hat keine LookTable.\n"
                    "Es ist bereits invariant.")
                return
        except Exception as e:
            messagebox.showerror("Fehler", f"DCP konnte nicht gelesen werden:\n{e}")
            return

        base = os.path.splitext(src)[0]
        dest = filedialog.asksaveasfilename(
            title="Invariantes DCP speichern als",
            initialfile=os.path.basename(base) + "_invariant.dcp",
            initialdir=os.path.dirname(src),
            defaultextension=".dcp",
            filetypes=[("DCP-Profil", "*.dcp")])
        if not dest:
            return
        try:
            make_invariant(profile)
            DCPWriter().write(dest, profile)
            messagebox.showinfo("Erfolg",
                f"LookTable in HueSatMap gemergt!\n\n"
                f"Profil: {profile.profile_name}\n"
                f"Datei: {os.path.basename(dest)}")
            self._update_status(f"Invariant: {os.path.basename(dest)}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Operation fehlgeschlagen:\n{e}")

    def _dcp_untwist(self):
        """DCP untwisten (Value-Dimension entfernen)."""
        if not HAS_DCP_XML:
            return
        src = filedialog.askopenfilename(
            title="DCP-Profil wählen",
            filetypes=[("DCP-Profil", "*.dcp"), ("Alle Dateien", "*.*")])
        if not src:
            return
        try:
            profile = DCPReader().read(src)
            has_twist = False
            if (profile.hue_sat_map_dims and profile.hue_sat_map_dims[2] > 1):
                has_twist = True
            if (profile.look_table_dims and profile.look_table_dims[2] > 1):
                has_twist = True
            if not has_twist:
                messagebox.showinfo("Info",
                    "Dieses Profil hat keine Hue Twists\n"
                    "(Value-Dimension ist bereits 1).")
                return
        except Exception as e:
            messagebox.showerror("Fehler", f"DCP konnte nicht gelesen werden:\n{e}")
            return

        base = os.path.splitext(src)[0]
        dest = filedialog.asksaveasfilename(
            title="Untwisted DCP speichern als",
            initialfile=os.path.basename(base) + "_untwist.dcp",
            initialdir=os.path.dirname(src),
            defaultextension=".dcp",
            filetypes=[("DCP-Profil", "*.dcp")])
        if not dest:
            return
        try:
            untwist(profile)
            DCPWriter().write(dest, profile)
            messagebox.showinfo("Erfolg",
                f"Hue Twists entfernt!\n\n"
                f"Profil: {profile.profile_name}\n"
                f"Datei: {os.path.basename(dest)}")
            self._update_status(f"Untwisted: {os.path.basename(dest)}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Operation fehlgeschlagen:\n{e}")

    # ── PGM → DNG Konvertierung ─────────────────────────────

    def _pgm_to_dng(self):
        """Dialog zur PGM → DNG Konvertierung."""
        if not HAS_DNG_WRITER:
            return

        pgm_path = filedialog.askopenfilename(
            title="PGM-Datei wählen",
            filetypes=[("PGM-Dateien", "*.pgm *.PGM"), ("Alle Dateien", "*.*")])
        if not pgm_path:
            return

        try:
            pgm = read_pgm(pgm_path)
        except Exception as e:
            messagebox.showerror("Fehler", f"PGM konnte nicht gelesen werden:\n{e}")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("PGM → DNG konvertieren")
        dlg.geometry("480x340")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = ttk.Frame(dlg, padding=15)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text=f"Datei: {os.path.basename(pgm_path)}",
                  font=('', 9, 'bold')).pack(anchor='w')
        ttk.Label(frame,
                  text=f"{pgm.width}×{pgm.height}, {pgm.bits_per_sample}-bit, "
                       f"maxval={pgm.max_val}").pack(anchor='w', pady=(2, 10))

        # CFA Pattern
        ttk.Label(frame, text="Bayer-Pattern:").pack(anchor='w')
        pattern_var = tk.StringVar(value="RGGB")
        pattern_frame = ttk.Frame(frame)
        pattern_frame.pack(anchor='w', pady=(2, 5))
        for pat in ["RGGB", "BGGR", "GRBG", "GBRG", "MONO"]:
            ttk.Radiobutton(pattern_frame, text=pat, value=pat,
                            variable=pattern_var).pack(side='left', padx=3)

        # Kameramodell
        ttk.Label(frame, text="Kameramodell:").pack(anchor='w', pady=(5, 0))
        camera_var = tk.StringVar(value=self.manual_camera_var.get() or "PGM Sensor")
        ttk.Entry(frame, textvariable=camera_var, width=40).pack(anchor='w', pady=(2, 5))

        # Weißabgleich
        ttk.Label(frame, text="Weißabgleich R,G,B (z.B. 0.47,1.0,0.63):").pack(anchor='w')
        wb_var = tk.StringVar()
        ttk.Entry(frame, textvariable=wb_var, width=40).pack(anchor='w', pady=(2, 5))

        # Black/White Level
        level_frame = ttk.Frame(frame)
        level_frame.pack(anchor='w', pady=(5, 10))
        ttk.Label(level_frame, text="Black-Level:").pack(side='left')
        black_var = tk.StringVar(value="0")
        ttk.Entry(level_frame, textvariable=black_var, width=8).pack(side='left', padx=(2, 15))
        ttk.Label(level_frame, text="White-Level:").pack(side='left')
        white_var = tk.StringVar(value=str(pgm.max_val))
        ttk.Entry(level_frame, textvariable=white_var, width=8).pack(side='left', padx=2)

        def do_convert():
            config = DNGConfig(
                camera_model=camera_var.get().strip() or "PGM Sensor",
                cfa_pattern=pattern_var.get(),
                black_level=int(black_var.get() or 0),
                white_level=int(white_var.get() or pgm.max_val),
            )

            # Weißabgleich parsen
            wb_text = wb_var.get().strip()
            if wb_text:
                try:
                    vals = [float(v.strip()) for v in wb_text.split(',')]
                    if len(vals) == 3:
                        mx = max(vals)
                        config.as_shot_neutral = tuple(v / mx for v in vals)
                except ValueError:
                    pass

            # Kamera-Farbmatrizen übernehmen
            cam_name = self.camera_var.get()
            if cam_name and cam_name in self.camera_map:
                cam = self.camera_map[cam_name]
                config.color_matrix_1 = cam.color_matrix_a
                config.color_matrix_2 = cam.color_matrix_d65

            base = os.path.splitext(pgm_path)[0]
            dng_path = filedialog.asksaveasfilename(
                title="DNG speichern als",
                initialfile=os.path.basename(base) + ".dng",
                initialdir=os.path.dirname(pgm_path),
                defaultextension=".dng",
                filetypes=[("DNG-Datei", "*.dng")],
                parent=dlg)
            if not dng_path:
                return

            try:
                DNGWriter().write(dng_path, pgm, config)
                dlg.destroy()
                messagebox.showinfo("Erfolg",
                    f"DNG erzeugt!\n\n"
                    f"{pgm.width}×{pgm.height}, {pgm.bits_per_sample}-bit\n"
                    f"Pattern: {config.cfa_pattern}\n"
                    f"Kamera: {config.camera_model}\n"
                    f"Datei: {os.path.basename(dng_path)}")
                self._update_status(f"DNG erzeugt: {os.path.basename(dng_path)}")
            except Exception as e:
                messagebox.showerror("Fehler", f"DNG-Erzeugung fehlgeschlagen:\n{e}",
                                     parent=dlg)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="DNG erzeugen…", command=do_convert).pack(side='right')
        ttk.Button(btn_frame, text="Abbrechen", command=dlg.destroy).pack(side='right', padx=(0, 5))

    # ── Batch Processing (Feature #9) ─────────────────────────

    def _open_batch_dialog(self):
        BatchDialog(self.root, self)

    # ── Recent Files ─────────────────────────────────────────

    def _add_to_recent(self, path: str):
        path = os.path.abspath(path)
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:MAX_RECENT]
        config = _load_config()
        config["recent_files"] = self.recent_files
        _save_config(config)
        self._update_recent_menu()

    def _update_recent_menu(self):
        if not hasattr(self, '_recent_menu'):
            return
        self._recent_menu.delete(0, 'end')
        for path in self.recent_files:
            if os.path.exists(path):
                label = f"{os.path.basename(path)}  ({os.path.dirname(path)})"
                self._recent_menu.add_command(
                    label=label,
                    command=lambda p=path: self._open_recent(p))
        if not self.recent_files:
            self._recent_menu.add_command(label="(keine)", state='disabled')

    def _open_recent(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.dcp':
            self._open_dcp(path)
        else:
            self._load_image_threaded(path)

    # ── Export All ───────────────────────────────────────────

    def _export_all(self):
        """Exportiert DCP + XMP + LUT + ICC auf einmal."""
        profile = self._build_dcp_profile()
        if not profile:
            return
        mix = self._get_mix_matrix()

        out_dir = filedialog.askdirectory(title="Exportordner wählen")
        if not out_dir:
            return

        base = f"{profile.camera_model.replace(' ', '_')}_{mix.name}"
        results = []
        errors = []

        # DCP
        try:
            p = os.path.join(out_dir, f"{base}.dcp")
            DCPWriter().write(p, profile)
            results.append(f"DCP: {os.path.basename(p)}")
        except Exception as e:
            errors.append(f"DCP: {e}")

        # XMP
        if HAS_XMP:
            try:
                from xmp_export import write_xmp_preset
                p = os.path.join(out_dir, f"{base}.xmp")
                write_xmp_preset(p, profile.profile_name, profile.camera_model)
                results.append(f"XMP: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"XMP: {e}")

        # LUT
        if HAS_LUT:
            try:
                p = os.path.join(out_dir, f"{base}.cube")
                lut = mix_matrix_to_lut(mix.matrix, size=33, title=mix.name)
                write_cube_lut(p, lut, title=f"DNG Channel Tool - {mix.name}")
                results.append(f"LUT: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"LUT: {e}")

        # ICC
        if HAS_ICC:
            try:
                p = os.path.join(out_dir, f"{base}.icc")
                channel_swap_to_icc(p, mix.matrix,
                                    f"DNG Channel Tool {mix.name}")
                results.append(f"ICC: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"ICC: {e}")

        msg = f"Exportiert nach: {out_dir}\n\n" + "\n".join(results)
        if errors:
            msg += "\n\nFehler:\n" + "\n".join(errors)
        messagebox.showinfo("Export abgeschlossen", msg)
        self._update_status(f"Export: {len(results)} Dateien in {out_dir}")

    # ── Canon/Sony Preset ────────────────────────────────────

    def _create_canon_sony_preset(self):
        """Canon Picture Style / Sony Creative Look Dialog."""
        if not HAS_CANON_SONY:
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Canon/Sony Preset erstellen")
        dlg.geometry("520x460")
        dlg.transient(self.root)
        dlg.grab_set()

        nb = ttk.Notebook(dlg)
        nb.pack(fill='both', expand=True, padx=10, pady=10)

        # ── Canon Tab ──
        canon_frame = ttk.Frame(nb, padding=10)
        nb.add(canon_frame, text="Canon Picture Style")

        ttk.Label(canon_frame, text="Basis-Stil:").grid(row=0, column=0, sticky='w')
        canon_style_var = tk.StringVar(value=CANON_BASE_STYLES[0])
        ttk.Combobox(canon_frame, textvariable=canon_style_var,
                     values=CANON_BASE_STYLES, state='readonly',
                     width=20).grid(row=0, column=1, sticky='w', padx=5)

        canon_params = {}
        canon_defs = [
            ("Schärfe", "sharpness", 0, 7, 3),
            ("Kontrast", "contrast", -4, 4, 0),
            ("Sättigung", "saturation", -4, 4, 0),
            ("Farbton", "color_tone", -4, 4, 0),
        ]
        for i, (label, key, lo, hi, default) in enumerate(canon_defs, start=1):
            ttk.Label(canon_frame, text=f"{label}:").grid(row=i, column=0, sticky='w')
            var = tk.IntVar(value=default)
            canon_params[key] = var
            ttk.Scale(canon_frame, from_=lo, to=hi, variable=var,
                      orient='horizontal', length=200).grid(row=i, column=1, padx=5)
            ttk.Label(canon_frame, textvariable=var, width=4).grid(row=i, column=2)

        def _canon_export_xmp():
            style = CanonPictureStyle(
                base_style=canon_style_var.get(),
                sharpness=canon_params['sharpness'].get(),
                contrast=canon_params['contrast'].get(),
                saturation=canon_params['saturation'].get(),
                color_tone=canon_params['color_tone'].get(),
            )
            path = filedialog.asksaveasfilename(
                title="Canon XMP speichern", defaultextension=".xmp",
                initialfile=f"Canon_{style.base_style}.xmp",
                filetypes=[("XMP-Preset", "*.xmp")], parent=dlg)
            if path:
                canon_to_xmp(style, path)
                messagebox.showinfo("Erfolg", f"Canon Preset exportiert:\n{os.path.basename(path)}", parent=dlg)

        ttk.Button(canon_frame, text="Als XMP exportieren…",
                   command=_canon_export_xmp).grid(row=6, column=0, columnspan=3, pady=15)

        # ── Sony Tab ──
        sony_frame = ttk.Frame(nb, padding=10)
        nb.add(sony_frame, text="Sony Creative Look")

        ttk.Label(sony_frame, text="Basis-Look:").grid(row=0, column=0, sticky='w')
        sony_look_var = tk.StringVar(value=SONY_BASE_LOOKS[0])
        ttk.Combobox(sony_frame, textvariable=sony_look_var,
                     values=SONY_BASE_LOOKS, state='readonly',
                     width=20).grid(row=0, column=1, sticky='w', padx=5)

        sony_params = {}
        sony_defs = [
            ("Kontrast", "contrast", -9, 9, 0),
            ("Lichter", "highlights", -9, 9, 0),
            ("Schatten", "shadows", -9, 9, 0),
            ("Verblassen", "fade", 0, 9, 0),
            ("Sättigung", "saturation", -9, 9, 0),
            ("Schärfe", "sharpness", -9, 9, 0),
            ("Klarheit", "clarity", -9, 9, 0),
        ]
        for i, (label, key, lo, hi, default) in enumerate(sony_defs, start=1):
            ttk.Label(sony_frame, text=f"{label}:").grid(row=i, column=0, sticky='w')
            var = tk.IntVar(value=default)
            sony_params[key] = var
            ttk.Scale(sony_frame, from_=lo, to=hi, variable=var,
                      orient='horizontal', length=200).grid(row=i, column=1, padx=5)
            ttk.Label(sony_frame, textvariable=var, width=4).grid(row=i, column=2)

        def _sony_export_xmp():
            look = SonyCreativeLook(
                base_look=sony_look_var.get(),
                contrast=sony_params['contrast'].get(),
                highlights=sony_params['highlights'].get(),
                shadows=sony_params['shadows'].get(),
                fade=sony_params['fade'].get(),
                saturation=sony_params['saturation'].get(),
                sharpness=sony_params['sharpness'].get(),
                clarity=sony_params['clarity'].get(),
            )
            path = filedialog.asksaveasfilename(
                title="Sony XMP speichern", defaultextension=".xmp",
                initialfile=f"Sony_{look.base_look}.xmp",
                filetypes=[("XMP-Preset", "*.xmp")], parent=dlg)
            if path:
                sony_to_xmp(look, path)
                messagebox.showinfo("Erfolg", f"Sony Preset exportiert:\n{os.path.basename(path)}", parent=dlg)

        ttk.Button(sony_frame, text="Als XMP exportieren…",
                   command=_sony_export_xmp).grid(row=9, column=0, columnspan=3, pady=15)


# ── CLI Mode ─────────────────────────────────────────────────

SWAP_ALIASES = {
    "RB": (2, 1, 0), "BR": (2, 1, 0),
    "RG": (1, 0, 2), "GR": (1, 0, 2),
    "GB": (0, 2, 1), "BG": (0, 2, 1),
    "RGB": (0, 1, 2),
    "GBR": (1, 2, 0),
    "BRG": (2, 0, 1),
    "BGR": (2, 1, 0),
    "GRB": (1, 0, 2),
    "RBG": (0, 2, 1),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DNG Channel Tool - Farbkanal-Tausch & Adobe DCP-Export",
        epilog="Ohne Argumente wird die GUI gestartet.")
    parser.add_argument("input", nargs="?", help="Eingabedatei (Bild oder DCP)")
    parser.add_argument("--swap", choices=sorted(set(SWAP_ALIASES.keys())),
                        help="Kanal-Tausch (z.B. RB, RG, GB, GBR, BRG)")
    parser.add_argument("--mix", metavar="MATRIX",
                        help="Mix-Matrix: 9 Komma-getrennte Werte")
    parser.add_argument("--camera", help="Kamera-Modellname für DCP")
    parser.add_argument("--profile-name", help="DCP-Profilname")
    parser.add_argument("--load-dcp", metavar="PATH", help="Basis-DCP-Profil laden")
    parser.add_argument("--export-dcp", metavar="PATH", help="DCP-Profil exportieren")
    parser.add_argument("--export-xmp", metavar="PATH", help="XMP-Preset exportieren")
    parser.add_argument("--export-lut", metavar="PATH", help="3D LUT exportieren")
    parser.add_argument("--export-icc", metavar="PATH", help="ICC-Profil exportieren")
    parser.add_argument("--export-image", metavar="PATH", help="Bild mit Swap speichern")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-Ausgabe")
    return parser.parse_args()


def cli_main(args):
    """Headless-Modus ohne GUI."""
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Mix-Matrix bestimmen
    if args.swap:
        perm = SWAP_ALIASES[args.swap]
        mapping = ChannelMapping.from_permutation(perm)
        mix = mapping.to_mix_matrix()
    elif args.mix:
        vals = [float(v) for v in args.mix.split(",")]
        if len(vals) != 9:
            print("FEHLER: --mix braucht genau 9 Werte", file=sys.stderr)
            sys.exit(1)
        mix = MixMatrix(matrix=np.array(vals).reshape(3, 3))
    else:
        mix = MixMatrix()

    # Basis-DCP laden
    loaded_dcp = None
    if args.load_dcp:
        loaded_dcp = DCPReader().read(args.load_dcp)
        print(f"DCP geladen: {loaded_dcp.profile_name} ({loaded_dcp.camera_model})")

    camera = args.camera or (loaded_dcp.camera_model if loaded_dcp else None)
    prof_name = args.profile_name or f"Channel Swap {mix.name}"

    # DCP exportieren
    if args.export_dcp:
        if not camera:
            print("FEHLER: --camera benötigt für DCP-Export", file=sys.stderr)
            sys.exit(1)
        if loaded_dcp:
            profile = DCPProfile(
                camera_model=camera, profile_name=prof_name,
                color_matrix_1=apply_to_color_matrix(loaded_dcp.color_matrix_1, mix),
                color_matrix_2=apply_to_color_matrix(loaded_dcp.color_matrix_2, mix)
                    if loaded_dcp.color_matrix_2 is not None else None,
                forward_matrix_1=apply_to_forward_matrix(loaded_dcp.forward_matrix_1, mix)
                    if loaded_dcp.forward_matrix_1 is not None else None,
                forward_matrix_2=apply_to_forward_matrix(loaded_dcp.forward_matrix_2, mix)
                    if loaded_dcp.forward_matrix_2 is not None else None,
                illuminant_1=loaded_dcp.illuminant_1,
                illuminant_2=loaded_dcp.illuminant_2,
                tone_curve_data=loaded_dcp.tone_curve_data,
                tone_curve_count=loaded_dcp.tone_curve_count,
            )
        else:
            # sRGB-Fallback
            identity = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ])
            cm = np.linalg.inv(identity)
            profile = DCPProfile(
                camera_model=camera, profile_name=prof_name,
                color_matrix_1=apply_to_color_matrix(cm, mix),
                illuminant_1=ILLUMINANT_D65)
        profile.copyright = "DNG Channel Tool"
        DCPWriter().write(args.export_dcp, profile)
        print(f"DCP exportiert: {args.export_dcp}")

    # XMP exportieren
    if args.export_xmp and HAS_XMP:
        from xmp_export import write_xmp_preset
        write_xmp_preset(args.export_xmp, prof_name, camera or "")
        print(f"XMP exportiert: {args.export_xmp}")

    # LUT exportieren
    if args.export_lut and HAS_LUT:
        lut = mix_matrix_to_lut(mix.matrix, size=33, title=mix.name)
        write_cube_lut(args.export_lut, lut,
                       title=f"DNG Channel Tool - {mix.name}")
        print(f"LUT exportiert: {args.export_lut}")

    # ICC exportieren
    if args.export_icc and HAS_ICC:
        channel_swap_to_icc(args.export_icc, mix.matrix,
                            f"DNG Channel Tool {mix.name}")
        print(f"ICC exportiert: {args.export_icc}")

    # Bild verarbeiten
    if args.input and args.export_image:
        img = np.array(Image.open(args.input))
        result = apply_to_image(img, mix)
        Image.fromarray(result).save(args.export_image)
        print(f"Bild gespeichert: {args.export_image}")


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()

    has_cli = any([args.export_dcp, args.export_xmp, args.export_lut,
                   args.export_icc, args.export_image])
    if has_cli:
        cli_main(args)
        return

    setup_logging()
    logger.info("DNG Channel Tool startet")

    root = tk.Tk()

    style = ttk.Style()
    try:
        style.theme_use('vista')
    except tk.TclError:
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

    app = ChannelToolApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
