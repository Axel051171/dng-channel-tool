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
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List
import numpy as np
from PIL import Image, ImageTk

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


WINDOW_TITLE = "DNG Channel Tool - Farbkanal-Tausch & Adobe DCP-Export"


# ── Custom Widgets ────────────────────────────────────────────

class AutocompleteCombobox(ttk.Combobox):
    """Combobox mit Suchfilter beim Tippen (Fix #4)."""

    def __init__(self, master, all_values=None, **kwargs):
        super().__init__(master, **kwargs)
        self._all_values = all_values or []
        self._is_filtering = False
        self.bind('<KeyRelease>', self._on_key)
        self.bind('<FocusIn>', self._on_focus_in)

    def set_all_values(self, values):
        self._all_values = list(values)
        self['values'] = self._all_values

    def _on_focus_in(self, event):
        if not self.get():
            self['values'] = self._all_values

    def _on_key(self, event):
        if event.keysym in ('Return', 'Tab', 'Escape', 'Up', 'Down'):
            return

        query = self.get().lower()
        if not query:
            self['values'] = self._all_values
            return

        # Filter: match anywhere in string
        filtered = [v for v in self._all_values if query in v.lower()]
        self['values'] = filtered

        if filtered and not self._is_filtering:
            self._is_filtering = True
            self.event_generate('<Down>')
            self._is_filtering = False


class HistogramWidget(tk.Canvas):
    """RGB-Histogramm-Widget (Feature #6)."""

    def __init__(self, master, height=100, **kwargs):
        super().__init__(master, height=height, bg='#1a1a1a', highlightthickness=0, **kwargs)
        self._height = height

    def update_histogram(self, image: Optional[np.ndarray]):
        self.delete('all')
        if image is None:
            return

        w = self.winfo_width()
        if w < 10:
            w = 300

        h = self._height
        colors = ['#ff3333', '#33cc33', '#3366ff']

        for ch, color in enumerate(colors):
            hist, _ = np.histogram(image[:, :, ch].ravel(), bins=256, range=(0, 256))
            if hist.max() == 0:
                continue

            hist_norm = hist.astype(float) / hist.max() * (h - 4)

            # Resample to canvas width
            x_scale = 256 / w
            points = []
            for x in range(w):
                bin_idx = min(255, int(x * x_scale))
                y = h - 2 - hist_norm[bin_idx]
                points.append((x, y))

            # Draw as filled polygon
            poly = [(0, h)] + points + [(w, h)]
            coords = []
            for px, py in poly:
                coords.extend([px, py])

            self.create_polygon(coords, fill='', outline=color, width=1,
                                stipple='gray50' if ch > 0 else '')


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
        file_menu.add_separator()
        file_menu.add_command(label="Batch-Verarbeitung…",
                              command=self._open_batch_dialog,
                              accelerator="Ctrl+B")
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        menubar.add_cascade(label="Datei", menu=file_menu)

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
        self.root.bind('<v>', lambda e: self._toggle_split())
        self.root.bind('<V>', lambda e: self._toggle_split())
        self.root.bind('<w>', lambda e: self._toggle_wb_picker() if HAS_WB else None)
        self.root.bind('<Control-l>', lambda e: self._open_preset_library()
                       if HAS_LIBRARY else None)

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

    def _on_drop(self, event):
        path = event.data.strip('{}')
        if path.lower().endswith('.dcp'):
            self._open_dcp(path)  # Fix #2: was _open_dcp_file
        else:
            self._load_image_threaded(path)

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

        self.preview_image = apply_ir_preset(self.original_image, preset)
        self._display_preview()
        self._update_histogram()
        self._update_status(f"IR-Preset: {preset_name} ({preset.description[:60]})")

    def _simulate_ir_filter(self, filter_type: str):
        """Simuliert einen IR-Filter auf dem aktuellen Bild."""
        if not HAS_IR or self.original_image is None:
            messagebox.showinfo("Info", "Bitte zuerst ein Bild laden.")
            return

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

    # ── Batch Processing (Feature #9) ─────────────────────────

    def _open_batch_dialog(self):
        BatchDialog(self.root, self)


class BatchDialog:
    """Batch-Verarbeitungsdialog (Feature #9)."""

    def __init__(self, parent, app: ChannelToolApp):
        self.app = app
        self.win = tk.Toplevel(parent)
        self.win.title("Batch-Verarbeitung")
        self.win.geometry("600x500")
        self.win.transient(parent)
        self.win.grab_set()

        self.files: List[str] = []

        # File list
        ttk.Label(self.win, text="Dateien:").pack(anchor='w', padx=10, pady=(10, 0))
        list_frame = ttk.Frame(self.win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Dateien hinzufügen…",
                   command=self._add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Ordner hinzufügen…",
                   command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Entfernen",
                   command=self._remove_selected).pack(side=tk.LEFT, padx=2)

        # Output settings
        out_frame = ttk.LabelFrame(self.win, text="  Ausgabe  ", padding=8)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(out_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Format:").pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value="PNG")
        ttk.Combobox(row1, textvariable=self.format_var, values=["PNG", "JPEG", "TIFF"],
                     state='readonly', width=10).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(out_frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Ordner:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.output_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="…", width=3,
                   command=self._choose_output_dir).pack(side=tk.LEFT)

        ttk.Label(out_frame, text="Aktueller Kanaltausch wird angewendet.",
                  foreground='gray').pack(anchor='w')

        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.win, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(self.win, text="")
        self.status_label.pack(anchor='w', padx=10)

        # Action buttons
        action_frame = ttk.Frame(self.win)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(action_frame, text="Starten",
                   command=self._start_batch).pack(side=tk.RIGHT, padx=2)
        ttk.Button(action_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=2)

    def _add_files(self):
        filetypes = [("Alle unterstützten",
                      "*.dng *.cr2 *.cr3 *.nef *.arw *.orf *.rw2 *.raf *.pef *.srw "
                      "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        paths = filedialog.askopenfilenames(title="Dateien hinzufügen", filetypes=filetypes)
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.file_list.insert(tk.END, os.path.basename(p))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Ordner auswählen")
        if not folder:
            return
        exts = {'.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.raf',
                '.pef', '.srw', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        for f in sorted(os.listdir(folder)):
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                full = os.path.join(folder, f)
                if full not in self.files:
                    self.files.append(full)
                    self.file_list.insert(tk.END, f)

    def _remove_selected(self):
        sel = list(self.file_list.curselection())
        for i in reversed(sel):
            self.file_list.delete(i)
            del self.files[i]

    def _choose_output_dir(self):
        d = filedialog.askdirectory(title="Ausgabeordner wählen")
        if d:
            self.output_var.set(d)

    def _start_batch(self):
        if not self.files:
            messagebox.showinfo("Info", "Keine Dateien ausgewählt.")
            return

        output_dir = self.output_var.get()
        if not output_dir:
            messagebox.showwarning("Ausgabeordner", "Bitte Ausgabeordner angeben.")
            return

        os.makedirs(output_dir, exist_ok=True)
        mix = self.app._get_mix_matrix()
        fmt = self.format_var.get()
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}
        out_ext = ext_map[fmt]

        total = len(self.files)

        def _run():
            raw_exts = {'.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2',
                        '.raf', '.pef', '.srw'}

            for idx, filepath in enumerate(self.files):
                try:
                    name = os.path.splitext(os.path.basename(filepath))[0]
                    self.win.after(0, lambda i=idx, n=name: (
                        self.status_label.config(text=f"Verarbeite: {n}…"),
                        self.progress_var.set((i / total) * 100)))

                    ext = os.path.splitext(filepath)[1].lower()
                    if ext in raw_exts and HAS_RAWPY:
                        with rawpy.imread(filepath) as raw:
                            image = raw.postprocess(use_camera_wb=True, output_bps=8,
                                                    no_auto_bright=True)
                    else:
                        image = np.array(Image.open(filepath).convert('RGB'))

                    result = apply_to_image(image, mix)
                    out_path = os.path.join(output_dir, name + out_ext)
                    Image.fromarray(result).save(out_path, quality=95)

                except Exception as e:
                    self.win.after(0, lambda n=os.path.basename(filepath), err=e:
                        self.status_label.config(text=f"Fehler bei {n}: {err}"))

            self.win.after(0, lambda: (
                self.progress_var.set(100),
                self.status_label.config(text=f"Fertig! {total} Dateien verarbeitet."),
                messagebox.showinfo("Batch fertig",
                    f"{total} Dateien verarbeitet.\nAusgabe: {output_dir}")))

        threading.Thread(target=_run, daemon=True).start()


class NEFExtractDialog:
    """Dialog zur Anzeige und zum Export extrahierter Nikon Picture Controls."""

    def __init__(self, parent, app: ChannelToolApp, pc: NikonPictureControl, nef_path: str):
        self.app = app
        self.pc = pc
        self.nef_path = nef_path

        self.win = tk.Toplevel(parent)
        self.win.title(f"Picture Control: {pc.name or 'Unbekannt'}")
        self.win.geometry("700x650")
        self.win.transient(parent)
        self.win.grab_set()

        # ── Header ──
        header = ttk.Frame(self.win)
        header.pack(fill=tk.X, padx=15, pady=(15, 5))

        tk.Label(header, text=pc.name or "Kein Name",
                 font=('Arial', 18, 'bold')).pack(anchor='w')
        tk.Label(header, text=f"Basis: {pc.base}  |  Version: {pc.version}  |  "
                 f"{'Monochrom' if pc.is_monochrome else 'Farbe'}",
                 font=('Arial', 10), fg='#666666').pack(anchor='w')

        # ── Notebook (Tabs) ──
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Tonkurve
        curve_frame = ttk.Frame(notebook, padding=10)
        notebook.add(curve_frame, text="Tonkurve")
        self._build_curve_tab(curve_frame)

        # Tab 2: Parameter
        param_frame = ttk.Frame(notebook, padding=10)
        notebook.add(param_frame, text="Parameter")
        self._build_param_tab(param_frame)

        # Tab 3: Weißabgleich & Sonstiges
        misc_frame = ttk.Frame(notebook, padding=10)
        notebook.add(misc_frame, text="Kamera-Einstellungen")
        self._build_misc_tab(misc_frame)

        # ── Action Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        ttk.Button(btn_frame, text="Als Lightroom-Preset speichern…",
                   command=self._save_xmp).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Direkt in Lightroom installieren",
                   command=self._install_to_lightroom).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Vorschaubild speichern…",
                   command=self._save_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

    def _build_curve_tab(self, parent):
        """Zeichnet die Tonkurve visuell."""
        if not self.pc.tone_curve:
            ttk.Label(parent, text="Keine Tonkurve gefunden.").pack()
            return

        # Curve canvas
        canvas = tk.Canvas(parent, width=300, height=300, bg='#1a1a1a',
                           highlightthickness=1, highlightbackground='#444')
        canvas.pack(pady=10)

        w, h = 300, 300
        margin = 20

        # Grid
        for i in range(5):
            x = margin + i * (w - 2 * margin) / 4
            y = margin + i * (h - 2 * margin) / 4
            canvas.create_line(x, margin, x, h - margin, fill='#333', dash=(2, 4))
            canvas.create_line(margin, y, w - margin, y, fill='#333', dash=(2, 4))

        # Diagonal (linear reference)
        canvas.create_line(margin, h - margin, w - margin, margin, fill='#555', width=1)

        # Draw curve
        points = self.pc.tone_curve
        canvas_points = []
        for x, y in points:
            cx = margin + x / 255 * (w - 2 * margin)
            cy = (h - margin) - y / 255 * (h - 2 * margin)
            canvas_points.append((cx, cy))

        # Draw curve line
        if len(canvas_points) >= 2:
            for i in range(len(canvas_points) - 1):
                x1, y1 = canvas_points[i]
                x2, y2 = canvas_points[i + 1]
                canvas.create_line(x1, y1, x2, y2, fill='#ffcc00', width=2,
                                   smooth=True)

        # Draw control points
        for cx, cy in canvas_points:
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                               fill='#ffcc00', outline='#ffffff')

        # Point list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.X, pady=5)

        ttk.Label(list_frame, text="Kontrollpunkte:",
                  font=('Arial', 10, 'bold')).pack(anchor='w')

        points_text = "  |  ".join(f"({x}, {y})" for x, y in points)
        ttk.Label(list_frame, text=points_text, wraplength=600,
                  foreground='#888888').pack(anchor='w')

    def _build_param_tab(self, parent):
        """Zeigt Picture Control Parameter."""
        params = [
            ("Schärfe", self.pc.sharpening),
            ("MidRange-Schärfe", self.pc.mid_range_sharpening),
            ("Klarheit", self.pc.clarity),
            ("Kontrast", self.pc.contrast),
            ("Helligkeit", self.pc.brightness),
            ("Sättigung", self.pc.saturation),
            ("Farbton (Hue)", self.pc.hue),
        ]

        for name, val in params:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3)

            ttk.Label(row, text=f"{name}:", width=20, anchor='w').pack(side=tk.LEFT)

            if val is None:
                ttk.Label(row, text="Auto", foreground='#888888').pack(side=tk.LEFT)
            else:
                # Visual bar
                bar_frame = ttk.Frame(row)
                bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

                bar = tk.Canvas(bar_frame, height=20, bg='#2a2a2a', highlightthickness=0)
                bar.pack(fill=tk.X, padx=(0, 10))

                bar.update_idletasks()
                bw = bar.winfo_width() or 200

                # Draw center line
                center = bw / 2
                bar.create_line(center, 0, center, 20, fill='#555')

                # Draw value bar
                bar_len = val / 128 * center
                color = '#44aaff' if val >= 0 else '#ff6644'
                if bar_len > 0:
                    bar.create_rectangle(center, 3, center + bar_len, 17, fill=color, outline='')
                else:
                    bar.create_rectangle(center + bar_len, 3, center, 17, fill=color, outline='')

                ttk.Label(row, text=f"{val:+d}", width=6).pack(side=tk.RIGHT)

        # Mono-specific
        if self.pc.is_monochrome:
            ttk.Separator(parent).pack(fill=tk.X, pady=10)
            ttk.Label(parent, text="Monochrom-Einstellungen:",
                      font=('Arial', 10, 'bold')).pack(anchor='w')

            if self.pc.filter_effect:
                ttk.Label(parent, text=f"  Filtereffekt: {self.pc.filter_effect}").pack(anchor='w')
            if self.pc.toning_effect:
                ttk.Label(parent,
                          text=f"  Tonung: {self.pc.toning_effect} "
                               f"(Sättigung: {self.pc.toning_saturation})").pack(anchor='w')

    def _build_misc_tab(self, parent):
        """Zeigt Kamera-Einstellungen."""
        settings = [
            ("Weißabgleich", self.pc.wb_mode),
            ("WB R-Koeffizient", f"{self.pc.wb_r_coeff:.4f}"),
            ("WB B-Koeffizient", f"{self.pc.wb_b_coeff:.4f}"),
            ("Farbraum", self.pc.color_space),
            ("Active D-Lighting", self.pc.active_d_lighting),
            ("Vignettierung", self.pc.vignette_control),
            ("High-ISO NR", self.pc.high_iso_nr),
        ]

        for name, val in settings:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{name}:", width=20, anchor='w',
                      font=('Arial', 10)).pack(side=tk.LEFT)
            ttk.Label(row, text=str(val), font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        ttk.Separator(parent).pack(fill=tk.X, pady=10)

        ttk.Label(parent, text=f"Quelldatei: {os.path.basename(self.nef_path)}",
                  foreground='#888888').pack(anchor='w')

        if self.pc.preview_data:
            ttk.Label(parent,
                      text=f"Eingebettete Vorschau: {len(self.pc.preview_data):,} Bytes",
                      foreground='#888888').pack(anchor='w')

    # ── Actions ──

    def _save_xmp(self):
        """Speichert als XMP-Preset."""
        preset_name = self.pc.name or "Nikon Preset"
        path = filedialog.asksaveasfilename(
            title="Lightroom-Preset speichern",
            defaultextension=".xmp",
            initialfile=f"{preset_name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")])
        if not path:
            return

        try:
            picture_control_to_xmp(self.pc, path, preset_name)
            self.app._update_status(f"Preset gespeichert: {path}")
            messagebox.showinfo("Erfolg",
                f"Lightroom-Preset gespeichert:\n{path}\n\n"
                f"In Lightroom importieren über:\n"
                f"Vorgaben → Rechtsklick → Vorgaben importieren")
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def _install_to_lightroom(self):
        """Installiert direkt in den Lightroom-Preset-Ordner."""
        preset_name = self.pc.name or "Nikon Preset"

        try:
            preset_dir = os.path.join(
                os.environ.get('APPDATA', ''),
                'Adobe', 'CameraRaw', 'Settings', 'Nikon Picture Controls')
            os.makedirs(preset_dir, exist_ok=True)

            xmp_path = os.path.join(preset_dir, f"{preset_name}.xmp")
            picture_control_to_xmp(self.pc, xmp_path, preset_name)

            self.app._update_status(f"Preset installiert: {preset_name}")
            messagebox.showinfo("Erfolg",
                f"Preset \"{preset_name}\" installiert!\n\n"
                f"Pfad: {xmp_path}\n\n"
                f"Starte Lightroom / Camera Raw neu,\n"
                f"dann findest du es unter:\n"
                f"Vorgaben → Nikon Picture Controls → {preset_name}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Installation fehlgeschlagen:\n{e}")

    def _save_preview(self):
        """Speichert das eingebettete Vorschaubild."""
        if not self.pc.preview_data:
            messagebox.showinfo("Info", "Kein Vorschaubild in der Datei eingebettet.")
            return

        base = os.path.splitext(os.path.basename(self.nef_path))[0]
        path = filedialog.asksaveasfilename(
            title="Vorschaubild speichern",
            defaultextension=".jpg",
            initialfile=f"{base}_preview.jpg",
            filetypes=[("JPEG", "*.jpg")])
        if not path:
            return

        try:
            save_preview(self.pc, path)
            self.app._update_status(f"Vorschau gespeichert: {path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")


class FujiRecipeDialog:
    """Dialog zum Eingeben und Konvertieren von Fujifilm-Rezepten."""

    def __init__(self, parent, app: ChannelToolApp):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Fujifilm-Rezept Konverter")
        self.win.geometry("800x650")
        self.win.transient(parent)
        self.win.grab_set()

        # ── Header ──
        header = ttk.Frame(self.win)
        header.pack(fill=tk.X, padx=15, pady=(10, 5))
        tk.Label(header, text="Fujifilm Film Simulation Rezept",
                 font=('Arial', 14, 'bold')).pack(anchor='w')
        tk.Label(header, text="Rezept-Text einfügen (z.B. von fujixweekly.com) "
                 "oder Beispiel laden",
                 fg='#888888').pack(anchor='w')

        # ── Main content ──
        pane = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Text input
        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        ttk.Label(left, text="Rezept-Text:").pack(anchor='w')
        self.recipe_text = tk.Text(left, width=40, height=20, font=('Consolas', 10))
        self.recipe_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Insert default example
        self.recipe_text.insert('1.0', EXAMPLE_RECIPES["Kodachrome 64"])

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Parsen & Vorschau",
                   command=self._parse_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Aus Zwischenablage",
                   command=self._paste_from_clipboard).pack(side=tk.LEFT, padx=2)

        # Example dropdown
        example_frame = ttk.Frame(left)
        example_frame.pack(fill=tk.X, pady=5)
        ttk.Label(example_frame, text="Beispiele:").pack(side=tk.LEFT)
        self.example_var = tk.StringVar()
        example_combo = ttk.Combobox(example_frame, textvariable=self.example_var,
                                      values=list(EXAMPLE_RECIPES.keys()),
                                      state='readonly', width=25)
        example_combo.pack(side=tk.LEFT, padx=5)
        example_combo.bind('<<ComboboxSelected>>', self._load_example)

        # Right: Preview
        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        ttk.Label(right, text="Erkannte Einstellungen:").pack(anchor='w')
        self.preview_text = tk.Text(right, width=35, height=15, font=('Consolas', 9),
                                     state='disabled', bg='#f5f5f5')
        self.preview_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Lightroom profile info
        self.profile_label = ttk.Label(right, text="", foreground='#0066cc',
                                        font=('Arial', 10, 'bold'))
        self.profile_label.pack(anchor='w', pady=3)

        # ── Export buttons ──
        export_frame = ttk.LabelFrame(self.win, text="  Export  ", padding=8)
        export_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(export_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Preset-Name:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value="Kodachrome 64")
        ttk.Entry(row1, textvariable=self.name_var, width=30).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(export_frame)
        row2.pack(fill=tk.X, pady=3)
        ttk.Button(row2, text="Als Lightroom-Preset speichern…",
                   command=self._export_xmp).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2, text="Direkt in Lightroom installieren",
                   command=self._install_lightroom).pack(side=tk.LEFT, padx=3)
        if HAS_NPC:
            ttk.Button(row2, text="Als Nikon Preset…",
                       command=self._export_nikon).pack(side=tk.LEFT, padx=3)

        ttk.Button(row2, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

        # Initial parse
        self._parse_preview()

    def _get_recipe(self) -> FujiRecipe:
        text = self.recipe_text.get('1.0', 'end')
        name = self.name_var.get()
        return parse_recipe(text, name)

    def _parse_preview(self):
        recipe = self._get_recipe()

        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', 'end')

        lines = [
            f"Film Simulation: {recipe.film_simulation}",
            f"Adobe-Profil: {recipe.adobe_profile}",
            f"Monochrom: {'Ja' if recipe.is_monochrome else 'Nein'}",
            "",
            f"Grain: {recipe.grain_roughness} {recipe.grain_size}",
            f"Color Chrome: {recipe.color_chrome_effect}",
            f"CC FX Blue: {recipe.color_chrome_fx_blue}",
            "",
            f"WB: {recipe.wb_mode}",
            f"WB Shift: R{recipe.wb_shift_red:+d} / B{recipe.wb_shift_blue:+d}",
            f"Dynamic Range: {recipe.dynamic_range}",
            "",
            f"Highlight: {recipe.highlight:+.1f}",
            f"Shadow: {recipe.shadow:+.1f}",
            f"Color: {recipe.color:+d}",
            f"Sharpness: {recipe.sharpness:+d}",
            f"Noise Reduction: {recipe.noise_reduction:+d}",
            f"Clarity: {recipe.clarity:+d}",
        ]

        if recipe.is_monochrome:
            lines.append(f"Mono WC: {recipe.mono_color_wc:+d}")
            lines.append(f"Mono MG: {recipe.mono_color_mg:+d}")

        self.preview_text.insert('1.0', "\n".join(lines))
        self.preview_text.config(state='disabled')

        self.profile_label.config(
            text=f"LR-Profil: {recipe.adobe_profile}")

    def _paste_from_clipboard(self):
        try:
            text = self.win.clipboard_get()
            self.recipe_text.delete('1.0', 'end')
            self.recipe_text.insert('1.0', text)
            self._parse_preview()
        except tk.TclError:
            pass

    def _load_example(self, event=None):
        name = self.example_var.get()
        if name in EXAMPLE_RECIPES:
            self.recipe_text.delete('1.0', 'end')
            self.recipe_text.insert('1.0', EXAMPLE_RECIPES[name])
            self.name_var.set(name)
            self._parse_preview()

    def _export_xmp(self):
        recipe = self._get_recipe()
        name = recipe.name or recipe.film_simulation
        path = filedialog.asksaveasfilename(
            title="Lightroom-Preset speichern",
            defaultextension=".xmp",
            initialfile=f"{name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")])
        if path:
            try:
                recipe_to_xmp(recipe, path)
                messagebox.showinfo("Erfolg",
                    f"Fuji-Rezept als Lightroom-Preset gespeichert:\n{path}\n\n"
                    f"Film Simulation: {recipe.film_simulation}\n"
                    f"Adobe-Profil: {recipe.adobe_profile}")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def _install_lightroom(self):
        recipe = self._get_recipe()
        try:
            xmp_path = install_recipe_to_lightroom(recipe)
            name = recipe.name or recipe.film_simulation
            messagebox.showinfo("Erfolg",
                f"Fuji-Rezept installiert!\n\n"
                f"Pfad: {xmp_path}\n\n"
                f"In Lightroom unter:\n"
                f"Vorgaben → Fujifilm Rezepte → {name}")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def _export_nikon(self):
        recipe = self._get_recipe()
        nikon_pc = recipe_to_nikon_pc(recipe)
        name = nikon_pc.name
        path = filedialog.asksaveasfilename(
            title="Nikon Preset speichern",
            defaultextension=".NP3",
            initialfile=f"{name}.NP3",
            filetypes=[("Nikon Picture Control", "*.NP3 *.NCP")])
        if path:
            try:
                fmt = "0300" if path.upper().endswith('.NP3') else "0100"
                write_npc(path, nikon_pc, fmt)
                messagebox.showinfo("Erfolg",
                    f"Fuji-Rezept als Nikon Preset gespeichert:\n{path}\n\n"
                    f"Fuji: {recipe.film_simulation} → Nikon: {nikon_pc.base}")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))


class StyleResultDialog:
    """Dialog: Zeigt extrahierten Bildstil und bietet Export-Optionen."""

    def __init__(self, parent, app: ChannelToolApp, style: ImageStyle):
        self.app = app
        self.style = style

        self.win = tk.Toplevel(parent)
        self.win.title(f"Bildstil: {style.name}")
        self.win.geometry("620x580")
        self.win.transient(parent)
        self.win.grab_set()

        # ── Header ──
        header = ttk.Frame(self.win)
        header.pack(fill=tk.X, padx=15, pady=(15, 5))
        tk.Label(header, text=style.name,
                 font=('Arial', 16, 'bold')).pack(anchor='w')

        tags = []
        if style.is_monochrome:
            tags.append("Monochrom")
        if style.is_high_contrast:
            tags.append("Hoher Kontrast")
        if style.is_faded:
            tags.append("Angehobene Schwarztöne")
        if style.temperature > 500:
            tags.append("Warm")
        elif style.temperature < -500:
            tags.append("Kalt")
        tag_str = " | ".join(tags) if tags else "Normal"
        tk.Label(header, text=tag_str, fg='#888888',
                 font=('Arial', 10)).pack(anchor='w')

        # ── Tonkurve ──
        curve_frame = ttk.LabelFrame(self.win, text="  Tonkurve  ", padding=8)
        curve_frame.pack(fill=tk.X, padx=10, pady=5)

        canvas = tk.Canvas(curve_frame, width=250, height=180, bg='#1a1a1a',
                           highlightthickness=1, highlightbackground='#444')
        canvas.pack(side=tk.LEFT, padx=5)

        w, h, m = 250, 180, 15
        canvas.create_line(m, h - m, w - m, m, fill='#444', dash=(2, 4))

        # Luminanz-Kurve
        if style.tone_curve:
            self._draw_curve(canvas, style.tone_curve, '#ffffff', w, h, m)
        # RGB-Kurven
        for curve, color in [(style.tone_curve_r, '#ff4444'),
                              (style.tone_curve_g, '#44cc44'),
                              (style.tone_curve_b, '#4488ff')]:
            if curve:
                self._draw_curve(canvas, curve, color, w, h, m)

        # Legende
        legend = ttk.Frame(curve_frame)
        legend.pack(side=tk.LEFT, padx=10)
        for color, name in [('#ffffff', 'Luminanz'), ('#ff4444', 'Rot'),
                             ('#44cc44', 'Grün'), ('#4488ff', 'Blau')]:
            row = ttk.Frame(legend)
            row.pack(anchor='w', pady=1)
            tk.Canvas(row, width=12, height=12, bg=color,
                      highlightthickness=0).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(row, text=name).pack(side=tk.LEFT)

        # ── Parameter ──
        param_frame = ttk.LabelFrame(self.win, text="  Erkannte Parameter  ", padding=8)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        params = [
            ("Kontrast", style.contrast, -100, 100),
            ("Helligkeit", style.brightness, -100, 100),
            ("Sättigung", style.saturation, -100, 100),
            ("Klarheit", style.clarity, -100, 100),
        ]

        for name, val, lo, hi in params:
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{name}:", width=12).pack(side=tk.LEFT)

            bar = tk.Canvas(row, height=16, bg='#2a2a2a', highlightthickness=0, width=200)
            bar.pack(side=tk.LEFT, padx=5)
            center = 100
            bar_len = val / 100 * 100
            color = '#44aaff' if val >= 0 else '#ff6644'
            if bar_len >= 0:
                bar.create_rectangle(center, 2, center + bar_len, 14, fill=color, outline='')
            else:
                bar.create_rectangle(center + bar_len, 2, center, 14, fill=color, outline='')
            bar.create_line(center, 0, center, 16, fill='#666')

            ttk.Label(row, text=f"{val:+d}", width=6).pack(side=tk.LEFT)

        extra_info = ttk.Frame(param_frame)
        extra_info.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(extra_info,
                  text=f"Schwarzpunkt: {style.black_point}  |  Weißpunkt: {style.white_point}",
                  foreground='#888888').pack(anchor='w')
        if style.source_file:
            ttk.Label(extra_info, text=f"Quelle: {style.source_file}",
                      foreground='#888888').pack(anchor='w')

        # ── Export Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Als Lightroom-Preset…",
                   command=self._export_xmp).pack(side=tk.LEFT, padx=3)
        if HAS_NPC:
            ttk.Button(btn_frame, text="Als Nikon Preset…",
                       command=self._export_nikon).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="In Lightroom installieren",
                   command=self._install_lightroom).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

    def _draw_curve(self, canvas, points, color, w, h, m):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            x1 = m + points[i][0] / 255 * (w - 2 * m)
            y1 = (h - m) - points[i][1] / 255 * (h - 2 * m)
            x2 = m + points[i + 1][0] / 255 * (w - 2 * m)
            y2 = (h - m) - points[i + 1][1] / 255 * (h - 2 * m)
            canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

    def _export_xmp(self):
        name = self.style.name or "Extrahierter Stil"
        path = filedialog.asksaveasfilename(
            title="Lightroom-Preset speichern",
            defaultextension=".xmp",
            initialfile=f"{name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")])
        if path:
            try:
                style_to_xmp(self.style, path)
                messagebox.showinfo("Erfolg", f"Preset gespeichert:\n{path}")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def _export_nikon(self):
        nikon_pc = style_to_nikon_pc(self.style)
        name = nikon_pc.name
        ext = ".NP3"
        path = filedialog.asksaveasfilename(
            title="Nikon Preset speichern",
            defaultextension=ext,
            initialfile=f"{name}{ext}",
            filetypes=[("Nikon Picture Control", "*.NP3 *.NCP")])
        if path:
            try:
                fmt = "0300" if path.upper().endswith('.NP3') else "0100"
                write_npc(path, nikon_pc, fmt)
                messagebox.showinfo("Erfolg",
                    f"Nikon Preset gespeichert:\n{path}\n\n"
                    f"Auf SD-Karte nach /NIKON/CUSTOMPC/ kopieren.")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def _install_lightroom(self):
        name = self.style.name or "Extrahierter Stil"
        try:
            preset_dir = os.path.join(
                os.environ.get('APPDATA', ''),
                'Adobe', 'CameraRaw', 'Settings', 'Extrahierte Stile')
            os.makedirs(preset_dir, exist_ok=True)
            xmp_path = os.path.join(preset_dir, f"{name}.xmp")
            style_to_xmp(self.style, xmp_path)
            messagebox.showinfo("Erfolg",
                f"Preset installiert!\n\n"
                f"Pfad: {xmp_path}\n\n"
                f"In Lightroom unter:\n"
                f"Vorgaben → Extrahierte Stile → {name}")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))


class NikonPresetCreatorDialog:
    """Dialog zum Erstellen und Bearbeiten von Nikon Picture Controls."""

    def __init__(self, parent, app: ChannelToolApp,
                 existing_pc: 'NikonPictureControlFile' = None):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Nikon Picture Control erstellen")
        self.win.geometry("750x700")
        self.win.transient(parent)
        self.win.grab_set()

        # Initialize from existing or default
        pc = existing_pc or NikonPictureControlFile()

        # ── Name & Base ──
        header = ttk.LabelFrame(self.win, text="  Grundeinstellungen  ", padding=10)
        header.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(header)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Preset-Name:", width=15).pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=pc.name)
        ttk.Entry(row1, textvariable=self.name_var, width=25).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(header)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Basis-Profil:", width=15).pack(side=tk.LEFT)
        self.base_var = tk.StringVar(value=pc.base)
        bases = list(BASE_PROFILES.values())
        ttk.Combobox(row2, textvariable=self.base_var, values=bases,
                     state='readonly', width=22).pack(side=tk.LEFT, padx=5)

        row3 = ttk.Frame(header)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Ausgabeformat:", width=15).pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value="NP3 (Z-Serie)")
        ttk.Combobox(row3, textvariable=self.format_var,
                     values=["NCP (D-SLR)", "NP3 (Z-Serie)"],
                     state='readonly', width=22).pack(side=tk.LEFT, padx=5)

        # ── Parameters ──
        param_frame = ttk.LabelFrame(self.win, text="  Parameter  ", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        self.param_scales = {}
        params = [
            ("Schärfe", "sharpening", pc.sharpening),
            ("Klarheit", "clarity", pc.clarity),
            ("Kontrast", "contrast", pc.contrast),
            ("Helligkeit", "brightness", pc.brightness),
            ("Sättigung", "saturation", pc.saturation),
            ("Farbton", "hue", pc.hue),
        ]

        for label, key, val in params:
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=f"{label}:", width=12).pack(side=tk.LEFT)

            auto_var = tk.BooleanVar(value=(val is None))
            scale_var = tk.IntVar(value=val if val is not None else 128)

            auto_cb = ttk.Checkbutton(row, text="Auto", variable=auto_var)
            auto_cb.pack(side=tk.LEFT, padx=3)

            val_label = ttk.Label(row, text="0", width=5)
            val_label.pack(side=tk.RIGHT, padx=5)

            scale = ttk.Scale(row, from_=0, to=255, variable=scale_var,
                              orient=tk.HORIZONTAL, length=250)
            scale.pack(side=tk.RIGHT, padx=5)

            def _update_label(v, lbl=val_label, sv=scale_var, av=auto_var):
                if av.get():
                    lbl.config(text="Auto")
                else:
                    lbl.config(text=f"{sv.get() - 128:+d}")

            scale_var.trace_add('write', lambda *a, f=_update_label: f(None))
            auto_var.trace_add('write', lambda *a, f=_update_label: f(None))
            _update_label(None)

            self.param_scales[key] = (scale_var, auto_var)

        # ── Tone Curve ──
        curve_frame = ttk.LabelFrame(self.win, text="  Tonkurve  ", padding=10)
        curve_frame.pack(fill=tk.X, padx=10, pady=5)

        self.curve_canvas = tk.Canvas(curve_frame, width=260, height=200,
                                       bg='#1a1a1a', highlightthickness=1,
                                       highlightbackground='#444')
        self.curve_canvas.pack(side=tk.LEFT, padx=5)

        curve_controls = ttk.Frame(curve_frame)
        curve_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(curve_controls, text="Punkte (x, y):").pack(anchor='w')
        self.curve_text = tk.Text(curve_controls, width=20, height=8, font=('Consolas', 9))
        self.curve_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Fill curve points
        curve_str = "\n".join(f"{x}, {y}" for x, y in pc.tone_curve)
        self.curve_text.insert('1.0', curve_str)

        ttk.Button(curve_controls, text="Kurve aktualisieren",
                   command=self._update_curve_display).pack(fill=tk.X)
        ttk.Button(curve_controls, text="Linear zurücksetzen",
                   command=self._reset_curve).pack(fill=tk.X, pady=2)

        self._update_curve_display()

        # ── Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Als NPC/NP3 speichern…",
                   command=self._save_npc).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Auf SD-Karte kopieren…",
                   command=self._save_to_sd).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Als Lightroom-Preset…",
                   command=self._save_as_lightroom).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

    def _get_pc(self) -> 'NikonPictureControlFile':
        """Build NikonPictureControlFile from current UI state."""
        pc = NikonPictureControlFile()
        pc.name = self.name_var.get()[:19]
        pc.base = self.base_var.get()

        for key, (scale_var, auto_var) in self.param_scales.items():
            val = None if auto_var.get() else scale_var.get()
            setattr(pc, key, val)

        pc.tone_curve = self._parse_curve_text()
        return pc

    def _parse_curve_text(self) -> list:
        """Parse tone curve points from text widget."""
        points = []
        text = self.curve_text.get('1.0', 'end').strip()
        for line in text.split('\n'):
            line = line.strip()
            if ',' in line:
                try:
                    parts = line.split(',')
                    x = int(parts[0].strip())
                    y = int(parts[1].strip())
                    points.append((max(0, min(255, x)), max(0, min(255, y))))
                except (ValueError, IndexError):
                    continue
        return points if points else [(0, 0), (255, 255)]

    def _update_curve_display(self):
        """Draw the tone curve on the canvas."""
        points = self._parse_curve_text()
        c = self.curve_canvas
        c.delete('all')
        w, h = 260, 200
        m = 15

        # Grid
        c.create_line(m, h - m, w - m, m, fill='#444', dash=(2, 4))
        for i in range(5):
            x = m + i * (w - 2 * m) / 4
            y = m + i * (h - 2 * m) / 4
            c.create_line(x, m, x, h - m, fill='#333', dash=(2, 4))
            c.create_line(m, y, w - m, y, fill='#333', dash=(2, 4))

        # Curve
        if len(points) >= 2:
            for i in range(len(points) - 1):
                x1 = m + points[i][0] / 255 * (w - 2 * m)
                y1 = (h - m) - points[i][1] / 255 * (h - 2 * m)
                x2 = m + points[i + 1][0] / 255 * (w - 2 * m)
                y2 = (h - m) - points[i + 1][1] / 255 * (h - 2 * m)
                c.create_line(x1, y1, x2, y2, fill='#ffcc00', width=2)

            for x, y in points:
                cx = m + x / 255 * (w - 2 * m)
                cy = (h - m) - y / 255 * (h - 2 * m)
                c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                              fill='#ffcc00', outline='white')

    def _reset_curve(self):
        self.curve_text.delete('1.0', 'end')
        self.curve_text.insert('1.0', "0, 0\n255, 255")
        self._update_curve_display()

    def _get_format_version(self) -> str:
        return "0300" if "NP3" in self.format_var.get() else "0100"

    def _get_extension(self) -> str:
        return ".NP3" if "NP3" in self.format_var.get() else ".NCP"

    def _save_npc(self):
        pc = self._get_pc()
        ext = self._get_extension()
        path = filedialog.asksaveasfilename(
            title="Nikon Preset speichern",
            defaultextension=ext,
            initialfile=f"{pc.name}{ext}",
            filetypes=[("Nikon Picture Control", f"*{ext}"), ("Alle Dateien", "*.*")])
        if not path:
            return

        try:
            write_npc(path, pc, self._get_format_version())
            self.app._update_status(f"Nikon Preset gespeichert: {path}")
            messagebox.showinfo("Erfolg",
                f"Nikon Picture Control gespeichert:\n{path}\n\n"
                f"Name: {pc.name}\nBasis: {pc.base}\n"
                f"Tonkurve: {len(pc.tone_curve)} Punkte\n\n"
                f"Auf SD-Karte kopieren nach:\n/NIKON/CUSTOMPC/\n\n"
                f"In der Kamera importieren:\n"
                f"Aufnahme-Menü → Bildstile verwalten\n"
                f"→ Laden/Speichern → Von Karte laden")
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def _save_to_sd(self):
        pc = self._get_pc()
        ext = self._get_extension()

        cards = find_sd_cards()
        if cards:
            card = cards[0]
        else:
            card = filedialog.askdirectory(title="SD-Karte / Laufwerk wählen")
            if not card:
                return

        try:
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), f"{pc.name}{ext}")
            write_npc(tmp, pc, self._get_format_version())
            dest = install_to_camera(tmp, card)
            os.unlink(tmp)

            messagebox.showinfo("Erfolg",
                f"Preset auf SD-Karte kopiert!\n\n"
                f"Pfad: {dest}\n\n"
                f"In der Kamera:\n"
                f"Aufnahme-Menü → Bildstile verwalten\n"
                f"→ Laden/Speichern → Von Karte laden")
        except Exception as e:
            messagebox.showerror("Fehler", f"Kopieren fehlgeschlagen:\n{e}")

    def _save_as_lightroom(self):
        """Speichert das Preset auch als Lightroom XMP."""
        if not HAS_NEF_EXTRACT:
            messagebox.showinfo("Info", "XMP-Export benötigt nef_extract Modul.")
            return

        pc = self._get_pc()
        lr = to_lightroom_values(pc)

        path = filedialog.asksaveasfilename(
            title="Lightroom-Preset speichern",
            defaultextension=".xmp",
            initialfile=f"{pc.name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")])
        if not path:
            return

        try:
            # Build a NikonPictureControl for the xmp converter
            from nef_extract import NikonPictureControl as NefPC
            nef_pc = NefPC()
            nef_pc.name = pc.name
            nef_pc.base = pc.base
            nef_pc.is_monochrome = pc.is_monochrome
            nef_pc.tone_curve = pc.tone_curve

            # Convert unsigned to signed for nef_extract
            def to_signed(v):
                return None if v is None else v - 128

            nef_pc.contrast = to_signed(pc.contrast)
            nef_pc.brightness = to_signed(pc.brightness)
            nef_pc.saturation = to_signed(pc.saturation)
            nef_pc.clarity = to_signed(pc.clarity)
            nef_pc.sharpening = to_signed(pc.sharpening)
            nef_pc.hue = to_signed(pc.hue)

            picture_control_to_xmp(nef_pc, path, pc.name)
            self.app._update_status(f"Lightroom-Preset gespeichert: {path}")
            messagebox.showinfo("Erfolg",
                f"Lightroom-Preset gespeichert:\n{path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"XMP-Export fehlgeschlagen:\n{e}")


class PresetLibraryDialog:
    """Preset-Bibliothek mit Suche und Filterung (#4)."""

    def __init__(self, parent, app: ChannelToolApp):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Preset-Bibliothek")
        self.win.geometry("900x600")
        self.win.transient(parent)

        self._update_status = app._update_status

        # Scan presets
        self.all_presets = scan_all_presets()

        # ── Search/Filter bar ──
        toolbar = ttk.Frame(self.win)
        toolbar.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(toolbar, text="Suche:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', lambda *a: self._refresh_list())
        ttk.Entry(toolbar, textvariable=self.search_var, width=25).pack(side=tk.LEFT, padx=5)

        ttk.Label(toolbar, text="Format:").pack(side=tk.LEFT, padx=(10, 0))
        self.format_var = tk.StringVar(value="Alle")
        formats = ["Alle", "XMP", "DCP", "NPC", "NP3", "NCP", "CUBE"]
        ttk.Combobox(toolbar, textvariable=self.format_var, values=formats,
                     state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        self.format_var.trace_add('write', lambda *a: self._refresh_list())

        ttk.Label(toolbar, text=f"{len(self.all_presets)} Presets gefunden",
                  foreground='gray').pack(side=tk.RIGHT)

        # ── List + Details ──
        pane = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: List
        list_frame = ttk.Frame(pane)
        pane.add(list_frame, weight=2)

        columns = ('name', 'format', 'category')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=20)
        self.tree.heading('name', text='Name')
        self.tree.heading('format', text='Format')
        self.tree.heading('category', text='Kategorie')
        self.tree.column('name', width=250)
        self.tree.column('format', width=60)
        self.tree.column('category', width=150)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        # Right: Details
        detail_frame = ttk.Frame(pane)
        pane.add(detail_frame, weight=1)

        self.detail_text = tk.Text(detail_frame, width=30, height=20,
                                    font=('Consolas', 9), state='disabled',
                                    bg='#f5f5f5', wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # ── Action Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Öffnen / Laden",
                   command=self._open_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Im Explorer anzeigen",
                   command=self._show_in_explorer).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

        self._refresh_list()

    def _refresh_list(self):
        self.tree.delete(*self.tree.get_children())

        query = self.search_var.get()
        fmt = self.format_var.get()
        fmt_filter = "" if fmt == "Alle" else fmt

        filtered = filter_presets(self.all_presets, query, fmt_filter)

        for entry in filtered:
            self.tree.insert('', 'end', values=(entry.name, entry.format, entry.category),
                             tags=(entry.filepath,))

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return

        item = self.tree.item(sel[0])
        filepath = item['tags'][0] if item['tags'] else ""

        # Find matching preset
        for p in self.all_presets:
            if p.filepath == filepath:
                info = get_preset_info(p)
                self.detail_text.config(state='normal')
                self.detail_text.delete('1.0', 'end')

                lines = [f"Name: {info.get('name', '')}",
                         f"Format: {info.get('format', '')}",
                         f"Pfad: {info.get('path', '')}",
                         f"Größe: {info.get('size', 0):,} Bytes",
                         f"Kategorie: {info.get('category', '')}"]

                for k, v in info.items():
                    if k not in ('name', 'format', 'path', 'size', 'category'):
                        lines.append(f"{k}: {v}")

                self.detail_text.insert('1.0', "\n".join(lines))
                self.detail_text.config(state='disabled')
                break

    def _open_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        filepath = item['tags'][0] if item['tags'] else ""
        if not filepath:
            return

        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.dcp':
            self.app._open_dcp(filepath)
        elif ext in ('.npc', '.np3', '.ncp') and HAS_NPC:
            pc = read_npc(filepath)
            NikonPresetCreatorDialog(self.win, self.app, pc)
        elif ext == '.xmp':
            messagebox.showinfo("Info",
                f"XMP-Preset: {os.path.basename(filepath)}\n\n"
                f"In Lightroom importieren über:\n"
                f"Vorgaben → Rechtsklick → Vorgaben importieren")

    def _show_in_explorer(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        filepath = item['tags'][0] if item['tags'] else ""
        if filepath and os.path.exists(filepath):
            os.startfile(os.path.dirname(filepath))


# ── Main ──────────────────────────────────────────────────────

def main():
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
