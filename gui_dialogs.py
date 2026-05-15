"""
Dialog-Fenster für DNG Channel Tool

Enthält alle sekundären Dialoge:
- BatchDialog: Batch-Verarbeitung mehrerer Dateien
- NEFExtractDialog: Nikon Picture Control Anzeige & Export
- FujiRecipeDialog: Fujifilm-Rezept Eingabe & Konvertierung
- StyleResultDialog: Bildstil-Analyse Ergebnisse & Export
- NikonPresetCreatorDialog: Nikon Picture Control Editor
- PresetLibraryDialog: Preset-Browser mit Suche
"""

import os
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, TYPE_CHECKING
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from gui_app import ChannelToolApp

logger = logging.getLogger(__name__)

# ── Conditional imports (same as gui_app.py) ──
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    from nef_extract import (
        extract_picture_control, picture_control_to_xmp,
        save_preview, NikonPictureControl,
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
    from preset_library import scan_all_presets, filter_presets, get_preset_info
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False

try:
    from lut_export import write_cube_lut, combined_lut, mix_matrix_to_lut
    HAS_LUT = True
except ImportError:
    HAS_LUT = False

try:
    from camera_presets import (
        CanonPictureStyle, SonyCreativeLook,
        canon_to_lightroom_xmp, sony_to_lightroom_xmp,
        canon_to_nikon_npc, sony_to_nikon_npc,
        canon_to_sony, sony_to_canon,
        write_canon_pf3, write_sony_look_xml,
        CANON_BASE_STYLES, SONY_BASE_LOOKS, SONY_LOOK_NAMES,
    )
    HAS_CANON_SONY = True
except ImportError:
    HAS_CANON_SONY = False

try:
    from xmp_export import write_xmp_preset
    HAS_XMP = True
except ImportError:
    HAS_XMP = False

try:
    from icc_export import channel_swap_to_icc
    HAS_ICC = True
except ImportError:
    HAS_ICC = False

try:
    from dcp_io import DCPProfile, DCPWriter
    HAS_DCP = True
except ImportError:
    HAS_DCP = False

from channel_swap import apply_to_image, MixMatrix


# ═══════════════════════════════════════════════════════════
#  BatchDialog
# ═══════════════════════════════════════════════════════════

class BatchDialog:
    """Batch-Verarbeitungsdialog."""

    def __init__(self, parent, app: 'ChannelToolApp'):
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
        logger.info("Batch-Verarbeitung: %d Dateien → %s", total, output_dir)

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
                    logger.debug("Batch: %s → %s", name, out_path)

                except Exception as e:
                    logger.warning("Batch-Fehler bei %s: %s", os.path.basename(filepath), e)
                    self.win.after(0, lambda n=os.path.basename(filepath), err=e:
                        self.status_label.config(text=f"Fehler bei {n}: {err}"))

            logger.info("Batch-Verarbeitung abgeschlossen: %d Dateien", total)
            self.win.after(0, lambda: (
                self.progress_var.set(100),
                self.status_label.config(text=f"Fertig! {total} Dateien verarbeitet."),
                messagebox.showinfo("Batch fertig",
                    f"{total} Dateien verarbeitet.\nAusgabe: {output_dir}")))

        threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════
#  NEFExtractDialog
# ═══════════════════════════════════════════════════════════

class NEFExtractDialog:
    """Dialog zur Anzeige und zum Export extrahierter Nikon Picture Controls."""

    def __init__(self, parent, app: 'ChannelToolApp', pc: 'NikonPictureControl', nef_path: str):
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

        canvas = tk.Canvas(parent, width=300, height=300, bg='#1a1a1a',
                           highlightthickness=1, highlightbackground='#444')
        canvas.pack(pady=10)

        w, h = 300, 300
        margin = 20

        for i in range(5):
            x = margin + i * (w - 2 * margin) / 4
            y = margin + i * (h - 2 * margin) / 4
            canvas.create_line(x, margin, x, h - margin, fill='#333', dash=(2, 4))
            canvas.create_line(margin, y, w - margin, y, fill='#333', dash=(2, 4))

        canvas.create_line(margin, h - margin, w - margin, margin, fill='#555', width=1)

        points = self.pc.tone_curve
        canvas_points = []
        for x, y in points:
            cx = margin + x / 255 * (w - 2 * margin)
            cy = (h - margin) - y / 255 * (h - 2 * margin)
            canvas_points.append((cx, cy))

        if len(canvas_points) >= 2:
            for i in range(len(canvas_points) - 1):
                x1, y1 = canvas_points[i]
                x2, y2 = canvas_points[i + 1]
                canvas.create_line(x1, y1, x2, y2, fill='#ffcc00', width=2,
                                   smooth=True)

        for cx, cy in canvas_points:
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                               fill='#ffcc00', outline='#ffffff')

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
                bar_frame = ttk.Frame(row)
                bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

                bar = tk.Canvas(bar_frame, height=20, bg='#2a2a2a', highlightthickness=0)
                bar.pack(fill=tk.X, padx=(0, 10))

                bar.update_idletasks()
                bw = bar.winfo_width() or 200

                center = bw / 2
                bar.create_line(center, 0, center, 20, fill='#555')

                bar_len = val / 128 * center
                color = '#44aaff' if val >= 0 else '#ff6644'
                if bar_len > 0:
                    bar.create_rectangle(center, 3, center + bar_len, 17, fill=color, outline='')
                else:
                    bar.create_rectangle(center + bar_len, 3, center, 17, fill=color, outline='')

                ttk.Label(row, text=f"{val:+d}", width=6).pack(side=tk.RIGHT)

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

    def _save_xmp(self):
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


# ═══════════════════════════════════════════════════════════
#  FujiRecipeDialog
# ═══════════════════════════════════════════════════════════

class FujiRecipeDialog:
    """Dialog zum Eingeben und Konvertieren von Fujifilm-Rezepten."""

    def __init__(self, parent, app: 'ChannelToolApp'):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Fujifilm-Rezept Konverter")
        self.win.geometry("800x650")
        self.win.transient(parent)
        self.win.grab_set()

        header = ttk.Frame(self.win)
        header.pack(fill=tk.X, padx=15, pady=(10, 5))
        tk.Label(header, text="Fujifilm Film Simulation Rezept",
                 font=('Arial', 14, 'bold')).pack(anchor='w')
        tk.Label(header, text="Rezept-Text einfügen (z.B. von fujixweekly.com) "
                 "oder Beispiel laden",
                 fg='#888888').pack(anchor='w')

        pane = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Text input
        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        ttk.Label(left, text="Rezept-Text:").pack(anchor='w')
        self.recipe_text = tk.Text(left, width=40, height=20, font=('Consolas', 10))
        self.recipe_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.recipe_text.insert('1.0', EXAMPLE_RECIPES["Kodachrome 64"])

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Parsen & Vorschau",
                   command=self._parse_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Aus Zwischenablage",
                   command=self._paste_from_clipboard).pack(side=tk.LEFT, padx=2)

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

        self.profile_label = ttk.Label(right, text="", foreground='#0066cc',
                                        font=('Arial', 10, 'bold'))
        self.profile_label.pack(anchor='w', pady=3)

        # Export buttons
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

        self._parse_preview()

    def _get_recipe(self) -> 'FujiRecipe':
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


# ═══════════════════════════════════════════════════════════
#  StyleResultDialog
# ═══════════════════════════════════════════════════════════

class StyleResultDialog:
    """Dialog: Zeigt extrahierten Bildstil und bietet Export-Optionen."""

    def __init__(self, parent, app: 'ChannelToolApp', style: 'ImageStyle'):
        self.app = app
        self.style = style

        self.win = tk.Toplevel(parent)
        self.win.title(f"Bildstil: {style.name}")
        self.win.geometry("620x580")
        self.win.transient(parent)
        self.win.grab_set()

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

        # Tonkurve
        curve_frame = ttk.LabelFrame(self.win, text="  Tonkurve  ", padding=8)
        curve_frame.pack(fill=tk.X, padx=10, pady=5)

        canvas = tk.Canvas(curve_frame, width=250, height=180, bg='#1a1a1a',
                           highlightthickness=1, highlightbackground='#444')
        canvas.pack(side=tk.LEFT, padx=5)

        w, h, m = 250, 180, 15
        canvas.create_line(m, h - m, w - m, m, fill='#444', dash=(2, 4))

        if style.tone_curve:
            self._draw_curve(canvas, style.tone_curve, '#ffffff', w, h, m)
        for curve, color in [(style.tone_curve_r, '#ff4444'),
                              (style.tone_curve_g, '#44cc44'),
                              (style.tone_curve_b, '#4488ff')]:
            if curve:
                self._draw_curve(canvas, curve, color, w, h, m)

        legend = ttk.Frame(curve_frame)
        legend.pack(side=tk.LEFT, padx=10)
        for color, name in [('#ffffff', 'Luminanz'), ('#ff4444', 'Rot'),
                             ('#44cc44', 'Grün'), ('#4488ff', 'Blau')]:
            row = ttk.Frame(legend)
            row.pack(anchor='w', pady=1)
            tk.Canvas(row, width=12, height=12, bg=color,
                      highlightthickness=0).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(row, text=name).pack(side=tk.LEFT)

        # Parameter
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

        # Export Buttons
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


# ═══════════════════════════════════════════════════════════
#  NikonPresetCreatorDialog
# ═══════════════════════════════════════════════════════════

class NikonPresetCreatorDialog:
    """Dialog zum Erstellen und Bearbeiten von Nikon Picture Controls."""

    def __init__(self, parent, app: 'ChannelToolApp',
                 existing_pc: 'NikonPictureControlFile' = None):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Nikon Picture Control erstellen")
        self.win.geometry("750x700")
        self.win.transient(parent)
        self.win.grab_set()

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
        pc = NikonPictureControlFile()
        pc.name = self.name_var.get()[:19]
        pc.base = self.base_var.get()

        for key, (scale_var, auto_var) in self.param_scales.items():
            val = None if auto_var.get() else scale_var.get()
            setattr(pc, key, val)

        pc.tone_curve = self._parse_curve_text()
        return pc

    def _parse_curve_text(self) -> list:
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
        points = self._parse_curve_text()
        c = self.curve_canvas
        c.delete('all')
        w, h = 260, 200
        m = 15

        c.create_line(m, h - m, w - m, m, fill='#444', dash=(2, 4))
        for i in range(5):
            x = m + i * (w - 2 * m) / 4
            y = m + i * (h - 2 * m) / 4
            c.create_line(x, m, x, h - m, fill='#333', dash=(2, 4))
            c.create_line(m, y, w - m, y, fill='#333', dash=(2, 4))

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
            from nef_extract import NikonPictureControl as NefPC
            nef_pc = NefPC()
            nef_pc.name = pc.name
            nef_pc.base = pc.base
            nef_pc.is_monochrome = pc.is_monochrome
            nef_pc.tone_curve = pc.tone_curve

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


# ═══════════════════════════════════════════════════════════
#  PresetLibraryDialog
# ═══════════════════════════════════════════════════════════

class PresetLibraryDialog:
    """Preset-Bibliothek mit Suche und Filterung."""

    def __init__(self, parent, app: 'ChannelToolApp'):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Preset-Bibliothek")
        self.win.geometry("900x600")
        self.win.transient(parent)

        self._update_status = app._update_status

        self.all_presets = scan_all_presets()

        # Search/Filter bar
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

        # List + Details
        pane = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

        detail_frame = ttk.Frame(pane)
        pane.add(detail_frame, weight=1)

        self.detail_text = tk.Text(detail_frame, width=30, height=20,
                                    font=('Consolas', 9), state='disabled',
                                    bg='#f5f5f5', wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # Action Buttons
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
            try:
                from platform_paths import open_in_file_manager
                open_in_file_manager(filepath)
            except ImportError:
                os.startfile(os.path.dirname(filepath))


# ═══════════════════════════════════════════════════════════
#  CanonSonyPresetDialog
# ═══════════════════════════════════════════════════════════

class CanonSonyPresetDialog:
    """Vollständiger Editor für Canon Picture Styles und Sony Creative Looks.

    Features:
    - Alle Parameter mit Slidern und Live-Wertanzeige
    - Basis-Preset laden (Presets als Schnellstart)
    - Tonkurven-Editor (Canon)
    - Export: XMP, PF3, Sony XML, Nikon NPC, LUT
    - Kreuz-Konvertierung Canon ↔ Sony ↔ Nikon
    """

    def __init__(self, parent, app: 'ChannelToolApp'):
        if not HAS_CANON_SONY:
            messagebox.showinfo("Info",
                "Canon/Sony Presets nicht verfügbar.\n"
                "Modul camera_presets.py fehlt.")
            return

        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Canon / Sony Preset erstellen")
        self.win.geometry("800x720")
        self.win.transient(parent)
        self.win.grab_set()

        nb = ttk.Notebook(self.win)
        nb.pack(fill='both', expand=True, padx=10, pady=(10, 5))

        # ── Canon Tab ──
        canon_frame = ttk.Frame(nb, padding=10)
        nb.add(canon_frame, text="Canon Picture Style")
        self._build_canon_tab(canon_frame)

        # ── Sony Tab ──
        sony_frame = ttk.Frame(nb, padding=10)
        nb.add(sony_frame, text="Sony Creative Look")
        self._build_sony_tab(sony_frame)

        # ── Unterer Button-Bereich ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

    # ── Canon Tab aufbauen ──

    def _build_canon_tab(self, parent):
        # ── Header: Name & Basis ──
        header = ttk.LabelFrame(parent, text="  Grundeinstellungen  ", padding=8)
        header.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(header)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Preset-Name:", width=15).pack(side=tk.LEFT)
        self.canon_name_var = tk.StringVar(value="Custom")
        ttk.Entry(row1, textvariable=self.canon_name_var, width=25).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(header)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Basis-Stil:", width=15).pack(side=tk.LEFT)
        self.canon_base_var = tk.StringVar(value="Standard")
        base_combo = ttk.Combobox(row2, textvariable=self.canon_base_var,
                                   values=CANON_BASE_STYLES, state='readonly', width=22)
        base_combo.pack(side=tk.LEFT, padx=5)
        base_combo.bind('<<ComboboxSelected>>', self._on_canon_base_changed)

        # ── Parameter ──
        param_frame = ttk.LabelFrame(parent, text="  Parameter  ", padding=8)
        param_frame.pack(fill=tk.X, pady=5)

        self.canon_params = {}
        canon_defs = [
            ("Schärfe",    "sharpness",  0, 10, 3),
            ("Feinheit",   "fineness",   1, 5,  3),
            ("Schwellenwert", "threshold", 1, 5,  2),
            ("Kontrast",   "contrast",   -4, 4, 0),
            ("Sättigung",  "saturation", -4, 4, 0),
            ("Farbton",    "color_tone", -4, 4, 0),
        ]

        for i, (label, key, lo, hi, default) in enumerate(canon_defs):
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=f"{label}:", width=15).pack(side=tk.LEFT)
            var = tk.IntVar(value=default)
            self.canon_params[key] = var

            val_label = ttk.Label(row, text=str(default), width=5, anchor='e')
            val_label.pack(side=tk.RIGHT, padx=5)

            scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                              orient='horizontal', length=250,
                              command=lambda v, lbl=val_label: lbl.config(text=str(int(float(v)))))
            scale.pack(side=tk.RIGHT, padx=5)

        # ── Monochrom-Optionen ──
        mono_frame = ttk.LabelFrame(parent, text="  Monochrom-Optionen  ", padding=8)
        mono_frame.pack(fill=tk.X, pady=5)

        row_f = ttk.Frame(mono_frame)
        row_f.pack(fill=tk.X, pady=2)
        ttk.Label(row_f, text="Filter-Effekt:", width=15).pack(side=tk.LEFT)
        self.canon_filter_var = tk.StringVar(value="Keiner")
        ttk.Combobox(row_f, textvariable=self.canon_filter_var,
                     values=["Keiner", "Yellow", "Orange", "Red", "Green"],
                     state='readonly', width=12).pack(side=tk.LEFT, padx=5)

        row_t = ttk.Frame(mono_frame)
        row_t.pack(fill=tk.X, pady=2)
        ttk.Label(row_t, text="Tonung:", width=15).pack(side=tk.LEFT)
        self.canon_toning_var = tk.StringVar(value="Keiner")
        ttk.Combobox(row_t, textvariable=self.canon_toning_var,
                     values=["Keiner", "Sepia", "Blue", "Purple", "Green"],
                     state='readonly', width=12).pack(side=tk.LEFT, padx=5)

        # ── Tonkurve ──
        curve_frame = ttk.LabelFrame(parent, text="  Tonkurve  ", padding=8)
        curve_frame.pack(fill=tk.X, pady=5)

        self.canon_curve_canvas = tk.Canvas(curve_frame, width=200, height=150,
                                             bg='#1a1a1a', highlightthickness=1,
                                             highlightbackground='#444')
        self.canon_curve_canvas.pack(side=tk.LEFT, padx=5)

        curve_ctrl = ttk.Frame(curve_frame)
        curve_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(curve_ctrl, text="Punkte (x, y):").pack(anchor='w')
        self.canon_curve_text = tk.Text(curve_ctrl, width=15, height=6, font=('Consolas', 9))
        self.canon_curve_text.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canon_curve_text.insert('1.0', "0, 0\n255, 255")

        ttk.Button(curve_ctrl, text="Aktualisieren",
                   command=self._update_canon_curve).pack(fill=tk.X)
        ttk.Button(curve_ctrl, text="Linear",
                   command=self._reset_canon_curve).pack(fill=tk.X, pady=2)
        self._update_canon_curve()

        # ── Export-Buttons ──
        export_frame = ttk.LabelFrame(parent, text="  Export  ", padding=8)
        export_frame.pack(fill=tk.X, pady=5)

        btn_row1 = ttk.Frame(export_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="Als Lightroom XMP…",
                   command=self._canon_export_xmp).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="Als Canon PF3…",
                   command=self._canon_export_pf3).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="Als Nikon NPC…",
                   command=self._canon_export_nikon).pack(side=tk.LEFT, padx=3)

        btn_row2 = ttk.Frame(export_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="→ Sony konvertieren",
                   command=self._canon_to_sony).pack(side=tk.LEFT, padx=3)
        if HAS_LUT:
            ttk.Button(btn_row2, text="Als 3D LUT…",
                       command=self._canon_export_lut).pack(side=tk.LEFT, padx=3)

    def _on_canon_base_changed(self, event=None):
        """Setzt Monochrom-Optionen je nach Basis-Stil."""
        is_mono = self.canon_base_var.get() == "Monochrome"
        state = 'readonly' if is_mono else 'disabled'
        # Hier könnten wir die Combo-States setzen, aber readonly/normal reicht

    def _get_canon_style(self) -> 'CanonPictureStyle':
        """Liest die aktuellen UI-Werte als CanonPictureStyle."""
        filter_eff = self.canon_filter_var.get()
        toning_eff = self.canon_toning_var.get()
        curve = self._parse_canon_curve()

        return CanonPictureStyle(
            name=self.canon_name_var.get() or "Custom",
            base_style=self.canon_base_var.get(),
            sharpness=self.canon_params['sharpness'].get(),
            fineness=self.canon_params['fineness'].get(),
            threshold=self.canon_params['threshold'].get(),
            contrast=self.canon_params['contrast'].get(),
            saturation=self.canon_params['saturation'].get(),
            color_tone=self.canon_params['color_tone'].get(),
            filter_effect=None if filter_eff == "Keiner" else filter_eff,
            toning_effect=None if toning_eff == "Keiner" else toning_eff,
            tone_curve=curve,
        )

    def _parse_canon_curve(self) -> list:
        points = []
        text = self.canon_curve_text.get('1.0', 'end').strip()
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

    def _update_canon_curve(self):
        points = self._parse_canon_curve()
        c = self.canon_curve_canvas
        c.delete('all')
        w, h = 200, 150
        m = 10

        c.create_line(m, h - m, w - m, m, fill='#444', dash=(2, 4))
        for i in range(5):
            x = m + i * (w - 2 * m) / 4
            y = m + i * (h - 2 * m) / 4
            c.create_line(x, m, x, h - m, fill='#333', dash=(2, 4))
            c.create_line(m, y, w - m, y, fill='#333', dash=(2, 4))

        if len(points) >= 2:
            for i in range(len(points) - 1):
                x1 = m + points[i][0] / 255 * (w - 2 * m)
                y1 = (h - m) - points[i][1] / 255 * (h - 2 * m)
                x2 = m + points[i + 1][0] / 255 * (w - 2 * m)
                y2 = (h - m) - points[i + 1][1] / 255 * (h - 2 * m)
                c.create_line(x1, y1, x2, y2, fill='#ff6644', width=2)

            for x, y in points:
                cx = m + x / 255 * (w - 2 * m)
                cy = (h - m) - y / 255 * (h - 2 * m)
                c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                              fill='#ff6644', outline='white')

    def _reset_canon_curve(self):
        self.canon_curve_text.delete('1.0', 'end')
        self.canon_curve_text.insert('1.0', "0, 0\n255, 255")
        self._update_canon_curve()

    def _canon_export_xmp(self):
        style = self._get_canon_style()
        path = filedialog.asksaveasfilename(
            title="Canon XMP exportieren", defaultextension=".xmp",
            initialfile=f"Canon_{style.name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")], parent=self.win)
        if path:
            canon_to_lightroom_xmp(style, path)
            self.app._update_status(f"Canon XMP: {os.path.basename(path)}")
            messagebox.showinfo("Erfolg",
                f"Canon Picture Style als Lightroom-Preset exportiert:\n"
                f"{os.path.basename(path)}\n\n"
                f"Stil: {style.base_style}\n"
                f"Kontrast: {style.contrast:+d}  Sättigung: {style.saturation:+d}",
                parent=self.win)

    def _canon_export_pf3(self):
        style = self._get_canon_style()
        path = filedialog.asksaveasfilename(
            title="Canon PF3 exportieren", defaultextension=".pf3",
            initialfile=f"{style.name}.pf3",
            filetypes=[("Canon Picture Style", "*.pf3")], parent=self.win)
        if path:
            write_canon_pf3(path, style)
            self.app._update_status(f"Canon PF3: {os.path.basename(path)}")
            messagebox.showinfo("Erfolg",
                f"Canon Picture Style gespeichert:\n{os.path.basename(path)}",
                parent=self.win)

    def _canon_export_nikon(self):
        style = self._get_canon_style()
        try:
            npc = canon_to_nikon_npc(style)
            path = filedialog.asksaveasfilename(
                title="Als Nikon NPC exportieren", defaultextension=".NP3",
                initialfile=f"{style.name}.NP3",
                filetypes=[("Nikon Picture Control", "*.NP3 *.NPC")],
                parent=self.win)
            if path:
                from npc_io import write_npc
                write_npc(path, npc, "0300")
                self.app._update_status(f"Nikon NPC: {os.path.basename(path)}")
                messagebox.showinfo("Erfolg",
                    f"Canon → Nikon konvertiert:\n{os.path.basename(path)}\n\n"
                    f"Basis: {npc.base}\nKontrast: {npc.contrast}  Sättigung: {npc.saturation}",
                    parent=self.win)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konvertierung fehlgeschlagen:\n{e}",
                                parent=self.win)

    def _canon_to_sony(self):
        """Konvertiert Canon → Sony und wechselt zum Sony-Tab."""
        style = self._get_canon_style()
        look = canon_to_sony(style)
        self._load_sony_values(look)
        # Tab wechseln
        nb = self.win.winfo_children()[0]  # Notebook
        if isinstance(nb, ttk.Notebook):
            nb.select(1)
        messagebox.showinfo("Konvertiert",
            f"Canon \"{style.name}\" → Sony \"{look.display_name}\"\n\n"
            f"Parameter wurden in den Sony-Tab übertragen.",
            parent=self.win)

    def _canon_export_lut(self):
        style = self._get_canon_style()
        path = filedialog.asksaveasfilename(
            title="Canon LUT exportieren", defaultextension=".cube",
            initialfile=f"Canon_{style.name}.cube",
            filetypes=[("3D LUT", "*.cube")], parent=self.win)
        if path:
            # Tonkurve + Kontrast/Sättigung als LUT
            from lut_export import combined_lut
            contrast_factor = 1.0 + style.contrast * 0.06
            sat_factor = 1.0 + style.saturation * 0.12
            lut = combined_lut(
                tone_curve=style.tone_curve if len(style.tone_curve) > 2 else None,
                saturation=sat_factor,
                monochrome=style.is_monochrome,
                size=33)
            write_cube_lut(path, lut, f"Canon {style.name}")
            messagebox.showinfo("Erfolg",
                f"Canon LUT exportiert:\n{os.path.basename(path)}",
                parent=self.win)

    # ── Sony Tab aufbauen ──

    def _build_sony_tab(self, parent):
        # ── Header: Name & Basis ──
        header = ttk.LabelFrame(parent, text="  Grundeinstellungen  ", padding=8)
        header.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(header)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Preset-Name:", width=15).pack(side=tk.LEFT)
        self.sony_name_var = tk.StringVar(value="Custom Look")
        ttk.Entry(row1, textvariable=self.sony_name_var, width=25).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(header)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Basis-Look:", width=15).pack(side=tk.LEFT)
        self.sony_base_var = tk.StringVar(value="ST")
        look_names = [f"{k} - {v.display_name}" for k, v in SONY_BASE_LOOKS.items()]
        base_combo = ttk.Combobox(row2, textvariable=self.sony_base_var,
                                   values=list(SONY_BASE_LOOKS.keys()),
                                   state='readonly', width=22)
        base_combo.pack(side=tk.LEFT, padx=5)
        base_combo.bind('<<ComboboxSelected>>', self._on_sony_base_changed)

        # Info-Label
        self.sony_info_label = ttk.Label(header, text="Standard", foreground='gray')
        self.sony_info_label.pack(anchor='w', pady=(2, 0))

        # ── Quick-Presets ──
        quick_frame = ttk.LabelFrame(parent, text="  Schnellauswahl  ", padding=5)
        quick_frame.pack(fill=tk.X, pady=5)

        btn_grid = ttk.Frame(quick_frame)
        btn_grid.pack(fill=tk.X)
        for i, (key, look) in enumerate(SONY_BASE_LOOKS.items()):
            ttk.Button(btn_grid, text=f"{key}",
                       command=lambda l=look: self._load_sony_values(l),
                       width=5).grid(row=0, column=i, padx=2, pady=2)

        # ── Parameter ──
        param_frame = ttk.LabelFrame(parent, text="  Parameter  ", padding=8)
        param_frame.pack(fill=tk.X, pady=5)

        self.sony_params = {}
        sony_defs = [
            ("Kontrast",      "contrast",        -9, 9, 0),
            ("Lichter",       "highlights",      -9, 9, 0),
            ("Schatten",      "shadows",         -9, 9, 0),
            ("Verblassen",    "fade",             0, 9, 0),
            ("Sättigung",     "saturation",      -9, 9, 0),
            ("Schärfe",       "sharpness",       -9, 9, 0),
            ("Schärfebereich", "sharpness_range", -9, 9, 0),
            ("Klarheit",      "clarity",         -9, 9, 0),
        ]

        for i, (label, key, lo, hi, default) in enumerate(sony_defs):
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=f"{label}:", width=15).pack(side=tk.LEFT)
            var = tk.IntVar(value=default)
            self.sony_params[key] = var

            val_label = ttk.Label(row, text=str(default), width=5, anchor='e')
            val_label.pack(side=tk.RIGHT, padx=5)

            scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                              orient='horizontal', length=250,
                              command=lambda v, lbl=val_label: lbl.config(text=str(int(float(v)))))
            scale.pack(side=tk.RIGHT, padx=5)

        # ── Export-Buttons ──
        export_frame = ttk.LabelFrame(parent, text="  Export  ", padding=8)
        export_frame.pack(fill=tk.X, pady=5)

        btn_row1 = ttk.Frame(export_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="Als Lightroom XMP…",
                   command=self._sony_export_xmp).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="Als Sony XML…",
                   command=self._sony_export_xml).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row1, text="Als Nikon NPC…",
                   command=self._sony_export_nikon).pack(side=tk.LEFT, padx=3)

        btn_row2 = ttk.Frame(export_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="→ Canon konvertieren",
                   command=self._sony_to_canon).pack(side=tk.LEFT, padx=3)
        if HAS_LUT:
            ttk.Button(btn_row2, text="Als 3D LUT…",
                       command=self._sony_export_lut).pack(side=tk.LEFT, padx=3)

    def _on_sony_base_changed(self, event=None):
        key = self.sony_base_var.get()
        if key in SONY_BASE_LOOKS:
            look = SONY_BASE_LOOKS[key]
            self._load_sony_values(look)

    def _load_sony_values(self, look: 'SonyCreativeLook'):
        """Lädt Sony Creative Look Werte in die UI."""
        self.sony_base_var.set(look.name)
        self.sony_name_var.set(look.display_name)
        self.sony_info_label.config(text=look.display_name)

        mapping = {
            'contrast': look.contrast,
            'highlights': look.highlights,
            'shadows': look.shadows,
            'fade': look.fade,
            'saturation': look.saturation,
            'sharpness': look.sharpness,
            'sharpness_range': look.sharpness_range,
            'clarity': look.clarity,
        }
        for key, val in mapping.items():
            if key in self.sony_params:
                self.sony_params[key].set(val)

    def _get_sony_look(self) -> 'SonyCreativeLook':
        """Liest die aktuellen UI-Werte als SonyCreativeLook."""
        return SonyCreativeLook(
            name=self.sony_base_var.get(),
            display_name=self.sony_name_var.get() or "Custom Look",
            contrast=self.sony_params['contrast'].get(),
            highlights=self.sony_params['highlights'].get(),
            shadows=self.sony_params['shadows'].get(),
            fade=self.sony_params['fade'].get(),
            saturation=self.sony_params['saturation'].get(),
            sharpness=self.sony_params['sharpness'].get(),
            sharpness_range=self.sony_params['sharpness_range'].get(),
            clarity=self.sony_params['clarity'].get(),
        )

    def _sony_export_xmp(self):
        look = self._get_sony_look()
        path = filedialog.asksaveasfilename(
            title="Sony XMP exportieren", defaultextension=".xmp",
            initialfile=f"Sony_{look.display_name}.xmp",
            filetypes=[("XMP Preset", "*.xmp")], parent=self.win)
        if path:
            sony_to_lightroom_xmp(look, path)
            self.app._update_status(f"Sony XMP: {os.path.basename(path)}")
            messagebox.showinfo("Erfolg",
                f"Sony Creative Look als Lightroom-Preset exportiert:\n"
                f"{os.path.basename(path)}\n\n"
                f"Look: {look.display_name}\n"
                f"Kontrast: {look.contrast:+d}  Sättigung: {look.saturation:+d}",
                parent=self.win)

    def _sony_export_xml(self):
        look = self._get_sony_look()
        path = filedialog.asksaveasfilename(
            title="Sony XML exportieren", defaultextension=".xml",
            initialfile=f"{look.display_name}.xml",
            filetypes=[("Sony Creative Look", "*.xml")], parent=self.win)
        if path:
            write_sony_look_xml(path, look)
            self.app._update_status(f"Sony XML: {os.path.basename(path)}")
            messagebox.showinfo("Erfolg",
                f"Sony Creative Look gespeichert:\n{os.path.basename(path)}",
                parent=self.win)

    def _sony_export_nikon(self):
        look = self._get_sony_look()
        try:
            npc = sony_to_nikon_npc(look)
            path = filedialog.asksaveasfilename(
                title="Als Nikon NPC exportieren", defaultextension=".NP3",
                initialfile=f"{look.display_name}.NP3",
                filetypes=[("Nikon Picture Control", "*.NP3 *.NPC")],
                parent=self.win)
            if path:
                from npc_io import write_npc
                write_npc(path, npc, "0300")
                self.app._update_status(f"Nikon NPC: {os.path.basename(path)}")
                messagebox.showinfo("Erfolg",
                    f"Sony → Nikon konvertiert:\n{os.path.basename(path)}\n\n"
                    f"Basis: {npc.base}\nKontrast: {npc.contrast}  Sättigung: {npc.saturation}",
                    parent=self.win)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konvertierung fehlgeschlagen:\n{e}",
                                parent=self.win)

    def _sony_to_canon(self):
        """Konvertiert Sony → Canon und wechselt zum Canon-Tab."""
        look = self._get_sony_look()
        style = sony_to_canon(look)

        # Canon-Werte in UI laden
        self.canon_name_var.set(style.name)
        self.canon_base_var.set(style.base_style)
        self.canon_params['sharpness'].set(style.sharpness)
        self.canon_params['fineness'].set(style.fineness)
        self.canon_params['threshold'].set(style.threshold)
        self.canon_params['contrast'].set(style.contrast)
        self.canon_params['saturation'].set(style.saturation)
        self.canon_params['color_tone'].set(style.color_tone)
        self.canon_filter_var.set(style.filter_effect or "Keiner")
        self.canon_toning_var.set(style.toning_effect or "Keiner")

        # Tab wechseln
        nb = self.win.winfo_children()[0]
        if isinstance(nb, ttk.Notebook):
            nb.select(0)
        messagebox.showinfo("Konvertiert",
            f"Sony \"{look.display_name}\" → Canon \"{style.name}\"\n\n"
            f"Parameter wurden in den Canon-Tab übertragen.",
            parent=self.win)

    def _sony_export_lut(self):
        look = self._get_sony_look()
        path = filedialog.asksaveasfilename(
            title="Sony LUT exportieren", defaultextension=".cube",
            initialfile=f"Sony_{look.display_name}.cube",
            filetypes=[("3D LUT", "*.cube")], parent=self.win)
        if path:
            from lut_export import combined_lut
            sat_factor = 1.0 + look.saturation * 0.05
            fade_curve = None
            if look.fade > 0:
                black_lift = int(look.fade * 255 / 36)
                fade_curve = [(0, black_lift), (64, 64 + black_lift // 2),
                              (128, 128), (192, 192), (255, 255)]
            lut = combined_lut(
                tone_curve=fade_curve,
                saturation=sat_factor,
                monochrome=look.is_monochrome,
                size=33)
            write_cube_lut(path, lut, f"Sony {look.display_name}")
            messagebox.showinfo("Erfolg",
                f"Sony LUT exportiert:\n{os.path.basename(path)}",
                parent=self.win)


# ═══════════════════════════════════════════════════════════
#  ExportAllDialog
# ═══════════════════════════════════════════════════════════

class ExportAllDialog:
    """Dialog für den kombinierten Export aller Formate.

    Ermöglicht es, DCP + XMP + LUT + ICC + Bild auf einmal
    in einen Ordner zu exportieren, mit Fortschrittsanzeige
    und individueller Auswahl der Formate.
    """

    def __init__(self, parent, app: 'ChannelToolApp'):
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title("Alles exportieren")
        self.win.geometry("550x520")
        self.win.transient(parent)
        self.win.grab_set()

        # ── Ausgabeordner ──
        dir_frame = ttk.LabelFrame(self.win, text="  Ausgabeordner  ", padding=8)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill=tk.X)
        self.dir_var = tk.StringVar()
        ttk.Entry(dir_row, textvariable=self.dir_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_row, text="…", width=3,
                   command=self._choose_dir).pack(side=tk.RIGHT, padx=(5, 0))

        # ── Format-Auswahl ──
        fmt_frame = ttk.LabelFrame(self.win, text="  Exportformate  ", padding=8)
        fmt_frame.pack(fill=tk.X, padx=10, pady=5)

        self.fmt_vars = {}
        formats = [
            ("dcp",  "DCP-Profil (Adobe Camera Raw / Lightroom)", True),
            ("xmp",  "XMP-Preset (Lightroom Develop-Vorgabe)", HAS_XMP),
            ("lut",  "3D LUT (.cube – Resolve, Premiere, Photoshop)", HAS_LUT),
            ("icc",  "ICC-Profil (Color Management)", HAS_ICC),
            ("image", "Bild mit Kanaltausch (PNG/JPEG/TIFF)", True),
        ]

        for key, label, available in formats:
            var = tk.BooleanVar(value=available and key != "image")
            self.fmt_vars[key] = var
            cb = ttk.Checkbutton(fmt_frame, text=label, variable=var)
            cb.pack(anchor='w', pady=1)
            if not available:
                cb.config(state='disabled')
                var.set(False)

        # ── Bild-Format (wenn Bild-Export gewählt) ──
        img_frame = ttk.LabelFrame(self.win, text="  Bild-Optionen  ", padding=8)
        img_frame.pack(fill=tk.X, padx=10, pady=5)

        img_row = ttk.Frame(img_frame)
        img_row.pack(fill=tk.X)
        ttk.Label(img_row, text="Bildformat:").pack(side=tk.LEFT)
        self.img_format_var = tk.StringVar(value="PNG")
        ttk.Combobox(img_row, textvariable=self.img_format_var,
                     values=["PNG", "JPEG", "TIFF"], state='readonly',
                     width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(img_frame, text="Nur verfügbar wenn ein Bild geladen ist.",
                  foreground='gray').pack(anchor='w', pady=(3, 0))

        # ── Dateiname-Vorschau ──
        name_frame = ttk.LabelFrame(self.win, text="  Dateinamen  ", padding=8)
        name_frame.pack(fill=tk.X, padx=10, pady=5)

        self.basename_var = tk.StringVar(value="export")
        name_row = ttk.Frame(name_frame)
        name_row.pack(fill=tk.X)
        ttk.Label(name_row, text="Basis-Name:").pack(side=tk.LEFT)
        ttk.Entry(name_row, textvariable=self.basename_var, width=35).pack(side=tk.LEFT, padx=5)

        self.preview_label = ttk.Label(name_frame, text="", foreground='gray',
                                        wraplength=500)
        self.preview_label.pack(anchor='w', pady=(5, 0))

        # Auto-Basename setzen
        self._auto_basename()
        self.basename_var.trace_add('write', lambda *a: self._update_preview())
        self._update_preview()

        # ── Fortschritt ──
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.win, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(self.win, text="")
        self.status_label.pack(anchor='w', padx=10)

        # ── Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Exportieren",
                   command=self._start_export).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

    def _choose_dir(self):
        d = filedialog.askdirectory(title="Exportordner wählen", parent=self.win)
        if d:
            self.dir_var.set(d)

    def _auto_basename(self):
        """Setzt automatisch einen sinnvollen Basisnamen."""
        cam = self.app.manual_camera_var.get().strip()
        mix = self.app._get_mix_matrix()
        if cam:
            self.basename_var.set(f"{cam.replace(' ', '_')}_{mix.name}")
        else:
            self.basename_var.set(f"ChannelTool_{mix.name}")

    def _update_preview(self):
        base = self.basename_var.get() or "export"
        files = []
        if self.fmt_vars['dcp'].get():
            files.append(f"{base}.dcp")
        if self.fmt_vars['xmp'].get():
            files.append(f"{base}.xmp")
        if self.fmt_vars['lut'].get():
            files.append(f"{base}.cube")
        if self.fmt_vars['icc'].get():
            files.append(f"{base}.icc")
        if self.fmt_vars['image'].get():
            ext = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}.get(
                self.img_format_var.get(), ".png")
            files.append(f"{base}{ext}")
        self.preview_label.config(text="Dateien: " + ", ".join(files) if files else "Keine Formate ausgewählt")

    def _start_export(self):
        out_dir = self.dir_var.get()
        if not out_dir:
            messagebox.showwarning("Fehler", "Bitte Ausgabeordner wählen.", parent=self.win)
            return

        selected = [k for k, v in self.fmt_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Fehler", "Bitte mindestens ein Format auswählen.", parent=self.win)
            return

        os.makedirs(out_dir, exist_ok=True)
        base = self.basename_var.get() or "export"
        mix = self.app._get_mix_matrix()

        results = []
        errors = []
        total = len(selected)
        step = 0

        def _progress(msg):
            nonlocal step
            step += 1
            pct = step / total * 100
            self.progress_var.set(pct)
            self.status_label.config(text=msg)
            self.win.update_idletasks()

        # DCP
        if 'dcp' in selected:
            _progress("Exportiere DCP-Profil…")
            try:
                profile = self.app._build_dcp_profile()
                if profile:
                    from dcp_io import DCPWriter
                    p = os.path.join(out_dir, f"{base}.dcp")
                    DCPWriter().write(p, profile)
                    results.append(f"DCP: {os.path.basename(p)}")
                else:
                    errors.append("DCP: Kein Kameramodell ausgewählt")
            except Exception as e:
                errors.append(f"DCP: {e}")

        # XMP
        if 'xmp' in selected:
            _progress("Exportiere XMP-Preset…")
            try:
                profile = self.app._build_dcp_profile()
                if profile:
                    p = os.path.join(out_dir, f"{base}.xmp")
                    write_xmp_preset(p, profile.profile_name, profile.camera_model)
                    results.append(f"XMP: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"XMP: {e}")

        # LUT
        if 'lut' in selected:
            _progress("Generiere 3D LUT…")
            try:
                p = os.path.join(out_dir, f"{base}.cube")
                lut = mix_matrix_to_lut(mix.matrix, size=33, title=mix.name)
                write_cube_lut(p, lut, title=f"DNG Channel Tool - {mix.name}")
                results.append(f"LUT: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"LUT: {e}")

        # ICC
        if 'icc' in selected:
            _progress("Generiere ICC-Profil…")
            try:
                p = os.path.join(out_dir, f"{base}.icc")
                channel_swap_to_icc(p, mix.matrix, f"DNG Channel Tool {mix.name}")
                results.append(f"ICC: {os.path.basename(p)}")
            except Exception as e:
                errors.append(f"ICC: {e}")

        # Bild
        if 'image' in selected:
            _progress("Speichere Bild…")
            if self.app.preview_image is not None:
                try:
                    ext_map = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}
                    ext = ext_map.get(self.img_format_var.get(), ".png")
                    p = os.path.join(out_dir, f"{base}{ext}")
                    Image.fromarray(self.app.preview_image).save(p, quality=95)
                    results.append(f"Bild: {os.path.basename(p)}")
                except Exception as e:
                    errors.append(f"Bild: {e}")
            else:
                errors.append("Bild: Kein Bild geladen")

        # Fertig
        self.progress_var.set(100)
        self.status_label.config(text=f"Fertig! {len(results)} Dateien exportiert.")

        msg = f"Exportiert nach:\n{out_dir}\n\n"
        if results:
            msg += "Erfolgreich:\n" + "\n".join(f"  {r}" for r in results)
        if errors:
            msg += "\n\nFehler:\n" + "\n".join(f"  {e}" for e in errors)

        messagebox.showinfo("Export abgeschlossen", msg, parent=self.win)


# ═══════════════════════════════════════════════════════════
#  ToneCurveDialog - Interaktiver Tonkurven-Editor
# ═══════════════════════════════════════════════════════════

class ToneCurveDialog:
    """Interaktiver Tonkurven-Editor mit Click-Drag Kontrollpunkten.

    Ermöglicht das Erstellen und Bearbeiten von Tonkurven
    durch Klicken und Ziehen von Punkten auf einem Canvas.
    """

    def __init__(self, parent, app: 'ChannelToolApp',
                 initial_curve=None, on_apply=None):
        self.app = app
        self.on_apply = on_apply

        self.win = tk.Toplevel(parent)
        self.win.title("Tonkurven-Editor")
        self.win.geometry("520x580")
        self.win.transient(parent)
        self.win.grab_set()

        # Kontrollpunkte (x, y) im Bereich 0-255
        self.points = list(initial_curve or [(0, 0), (64, 64), (128, 128),
                                              (192, 192), (255, 255)])
        self._dragging_idx = None

        # ── Canvas ──
        canvas_frame = ttk.LabelFrame(self.win, text="  Tonkurve  ", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, width=400, height=400,
                                 bg='#1a1a1a', highlightthickness=1,
                                 highlightbackground='#555')
        self.canvas.pack(padx=5, pady=5)

        self.canvas.bind('<ButtonPress-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)
        self.canvas.bind('<Button-3>', self._on_right_click)

        # ── Info ──
        info = ttk.Label(self.win,
                          text="Klicken: Punkt hinzufügen  |  Ziehen: Punkt verschieben  |  "
                               "Rechtsklick: Punkt entfernen",
                          foreground='gray')
        info.pack(padx=10, pady=(0, 5))

        # ── Presets ──
        preset_frame = ttk.LabelFrame(self.win, text="  Vorlagen  ", padding=5)
        preset_frame.pack(fill=tk.X, padx=10, pady=5)

        presets = [
            ("Linear", [(0, 0), (255, 255)]),
            ("S-Kurve", [(0, 0), (64, 48), (128, 128), (192, 208), (255, 255)]),
            ("Kontrast+", [(0, 0), (48, 20), (128, 128), (208, 236), (255, 255)]),
            ("Aufhellen", [(0, 0), (128, 160), (255, 255)]),
            ("Abdunkeln", [(0, 0), (128, 96), (255, 255)]),
            ("Faded", [(0, 30), (64, 74), (128, 128), (192, 192), (255, 235)]),
            ("Filmisch", [(0, 15), (32, 40), (128, 135), (224, 230), (255, 245)]),
        ]
        btn_row = ttk.Frame(preset_frame)
        btn_row.pack(fill=tk.X)
        for i, (name, pts) in enumerate(presets):
            ttk.Button(btn_row, text=name, width=9,
                       command=lambda p=pts: self._load_preset(p)
                       ).grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky='ew')
        for col in range(4):
            btn_row.columnconfigure(col, weight=1)

        # ── Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Anwenden",
                   command=self._apply).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btn_frame, text="Vorschau",
                   command=self._preview).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btn_frame, text="Schließen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

        self._draw()

    def _canvas_to_point(self, cx, cy):
        """Canvas-Koordinaten → Kurven-Koordinaten (0-255)."""
        m = 20
        w = self.canvas.winfo_width() - 2 * m
        h = self.canvas.winfo_height() - 2 * m
        x = max(0, min(255, int((cx - m) / w * 255)))
        y = max(0, min(255, int((1.0 - (cy - m) / h) * 255)))
        return x, y

    def _point_to_canvas(self, x, y):
        """Kurven-Koordinaten (0-255) → Canvas-Koordinaten."""
        m = 20
        w = self.canvas.winfo_width() - 2 * m
        h = self.canvas.winfo_height() - 2 * m
        cx = m + x / 255 * w
        cy = m + (1.0 - y / 255) * h
        return cx, cy

    def _draw(self):
        """Zeichnet die Kurve und alle Kontrollpunkte."""
        c = self.canvas
        c.delete('all')
        cw = c.winfo_width() or 400
        ch = c.winfo_height() or 400
        m = 20

        # Gitterlinien
        for i in range(5):
            x = m + i * (cw - 2 * m) / 4
            y = m + i * (ch - 2 * m) / 4
            c.create_line(x, m, x, ch - m, fill='#333', dash=(2, 4))
            c.create_line(m, y, cw - m, y, fill='#333', dash=(2, 4))

        # Diagonale (linear)
        c.create_line(m, ch - m, cw - m, m, fill='#444', dash=(3, 3))

        # Kurve zeichnen
        if len(self.points) >= 2:
            sorted_pts = sorted(self.points, key=lambda p: p[0])
            for i in range(len(sorted_pts) - 1):
                x1, y1 = self._point_to_canvas(*sorted_pts[i])
                x2, y2 = self._point_to_canvas(*sorted_pts[i + 1])
                c.create_line(x1, y1, x2, y2, fill='#ffcc00', width=2, smooth=True)

            # Kontrollpunkte
            for i, (px, py) in enumerate(sorted_pts):
                cx, cy = self._point_to_canvas(px, py)
                r = 5
                c.create_oval(cx - r, cy - r, cx + r, cy + r,
                              fill='#ffcc00', outline='white', width=1)
                # Wert-Label
                c.create_text(cx, cy - 12, text=f"({px},{py})",
                              fill='#aaa', font=('Consolas', 7))

    def _find_nearest_point(self, cx, cy, threshold=15):
        """Findet den nächsten Punkt innerhalb des Schwellenwerts."""
        for i, (px, py) in enumerate(self.points):
            pcx, pcy = self._point_to_canvas(px, py)
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < threshold:
                return i
        return None

    def _on_click(self, event):
        idx = self._find_nearest_point(event.x, event.y)
        if idx is not None:
            self._dragging_idx = idx
        else:
            # Neuen Punkt hinzufügen
            x, y = self._canvas_to_point(event.x, event.y)
            self.points.append((x, y))
            self.points.sort(key=lambda p: p[0])
            self._draw()

    def _on_drag(self, event):
        if self._dragging_idx is not None:
            x, y = self._canvas_to_point(event.x, event.y)
            self.points[self._dragging_idx] = (x, y)
            self._draw()

    def _on_release(self, event):
        if self._dragging_idx is not None:
            self._dragging_idx = None
            self.points.sort(key=lambda p: p[0])
            self._draw()

    def _on_double_click(self, event):
        """Doppelklick: Punkt hinzufügen."""
        x, y = self._canvas_to_point(event.x, event.y)
        self.points.append((x, y))
        self.points.sort(key=lambda p: p[0])
        self._draw()

    def _on_right_click(self, event):
        """Rechtsklick: Nächsten Punkt entfernen (min 2 Punkte)."""
        if len(self.points) <= 2:
            return
        idx = self._find_nearest_point(event.x, event.y)
        if idx is not None:
            del self.points[idx]
            self._draw()

    def _load_preset(self, points):
        self.points = list(points)
        self._draw()

    def _apply_curve_to_image(self, image: np.ndarray) -> np.ndarray:
        """Wendet die Tonkurve auf ein Bild an."""
        sorted_pts = sorted(self.points, key=lambda p: p[0])
        xs = [p[0] for p in sorted_pts]
        ys = [p[1] for p in sorted_pts]

        # Lookup-Table durch lineare Interpolation erstellen
        lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
        return lut[image]

    def _preview(self):
        """Zeigt Vorschau der Kurve auf dem aktuellen Bild."""
        if self.app.original_image is None:
            return
        mix = self.app._get_mix_matrix()
        from channel_swap import apply_to_image
        swapped = apply_to_image(self.app.original_image, mix)
        self.app.preview_image = self._apply_curve_to_image(swapped)
        self.app._display_preview()
        self.app._update_histogram()

    def _apply(self):
        """Wendet die Kurve an und schließt den Dialog."""
        if self.app.original_image is not None:
            self.app._push_undo("Tonkurve")
            self._preview()
        if self.on_apply:
            self.on_apply(list(self.points))
        self.win.destroy()


# ═══════════════════════════════════════════════════════════
#  CropDialog - Interaktives Zuschneiden
# ═══════════════════════════════════════════════════════════

class CropDialog:
    """Interaktives Crop-Werkzeug mit Seitenverhältnis-Lock.

    Zeigt das Bild mit einem ziehbaren Crop-Rechteck an.
    """

    def __init__(self, parent, app: 'ChannelToolApp'):
        self.app = app
        if app.original_image is None:
            messagebox.showinfo("Info", "Kein Bild geladen.", parent=parent)
            return

        self.image = app.original_image
        self.ih, self.iw = self.image.shape[:2]

        self.win = tk.Toplevel(parent)
        self.win.title("Bild zuschneiden")
        self.win.geometry("800x650")
        self.win.transient(parent)
        self.win.grab_set()

        # Crop-Bereich (relativ, 0.0-1.0)
        self.crop_x1 = 0.1
        self.crop_y1 = 0.1
        self.crop_x2 = 0.9
        self.crop_y2 = 0.9
        self._drag_mode = None
        self._drag_start = None

        # ── Optionen ──
        opt_frame = ttk.Frame(self.win)
        opt_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(opt_frame, text="Seitenverhältnis:").pack(side=tk.LEFT)
        self.ratio_var = tk.StringVar(value="Frei")
        ratios = ["Frei", "1:1", "4:3", "3:2", "16:9", "5:4", "Original"]
        ttk.Combobox(opt_frame, textvariable=self.ratio_var,
                     values=ratios, state='readonly', width=10).pack(side=tk.LEFT, padx=5)

        self.size_label = ttk.Label(opt_frame, text="", foreground='gray')
        self.size_label.pack(side=tk.RIGHT, padx=10)

        # ── Canvas ──
        self.canvas = tk.Canvas(self.win, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', lambda e: self._draw())

        self.photo = None

        # ── Buttons ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Zuschneiden",
                   command=self._apply).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btn_frame, text="Zurücksetzen",
                   command=self._reset).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btn_frame, text="Abbrechen",
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=3)

        self.win.after(50, self._draw)

    def _draw(self):
        """Zeichnet Bild + Crop-Overlay."""
        c = self.canvas
        c.delete('all')
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Bild skalieren
        scale = min(cw / self.iw, ch / self.ih)
        dw = int(self.iw * scale)
        dh = int(self.ih * scale)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2

        self._ox, self._oy = ox, oy
        self._dw, self._dh = dw, dh
        self._scale = scale

        img = Image.fromarray(self.image)
        img = img.resize((max(1, dw), max(1, dh)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        c.create_image(ox, oy, image=self.photo, anchor=tk.NW)

        # Crop-Rechteck
        cx1 = ox + int(self.crop_x1 * dw)
        cy1 = oy + int(self.crop_y1 * dh)
        cx2 = ox + int(self.crop_x2 * dw)
        cy2 = oy + int(self.crop_y2 * dh)

        # Dunkles Overlay außerhalb des Crops
        c.create_rectangle(ox, oy, cx1, oy + dh, fill='black', stipple='gray50', outline='')
        c.create_rectangle(cx2, oy, ox + dw, oy + dh, fill='black', stipple='gray50', outline='')
        c.create_rectangle(cx1, oy, cx2, cy1, fill='black', stipple='gray50', outline='')
        c.create_rectangle(cx1, cy2, cx2, oy + dh, fill='black', stipple='gray50', outline='')

        # Crop-Rahmen
        c.create_rectangle(cx1, cy1, cx2, cy2, outline='#ffffff', width=2)

        # Drittel-Linien
        for i in range(1, 3):
            x = cx1 + (cx2 - cx1) * i / 3
            y = cy1 + (cy2 - cy1) * i / 3
            c.create_line(x, cy1, x, cy2, fill='#ffffff', dash=(3, 3))
            c.create_line(cx1, y, cx2, y, fill='#ffffff', dash=(3, 3))

        # Griff-Quadrate an Ecken
        for x, y in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]:
            c.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill='white', outline='#333')

        # Größe anzeigen
        crop_w = int((self.crop_x2 - self.crop_x1) * self.iw)
        crop_h = int((self.crop_y2 - self.crop_y1) * self.ih)
        self.size_label.config(text=f"{crop_w} x {crop_h} px")

    def _canvas_to_rel(self, cx, cy):
        """Canvas → relative Position (0-1) im Bild."""
        rx = max(0, min(1, (cx - self._ox) / self._dw))
        ry = max(0, min(1, (cy - self._oy) / self._dh))
        return rx, ry

    def _on_press(self, event):
        rx, ry = self._canvas_to_rel(event.x, event.y)
        # Prüfe ob Ecke oder Kante gezogen wird
        tol = 0.03
        near_x1 = abs(rx - self.crop_x1) < tol
        near_x2 = abs(rx - self.crop_x2) < tol
        near_y1 = abs(ry - self.crop_y1) < tol
        near_y2 = abs(ry - self.crop_y2) < tol

        if near_x1 and near_y1:
            self._drag_mode = 'tl'
        elif near_x2 and near_y1:
            self._drag_mode = 'tr'
        elif near_x1 and near_y2:
            self._drag_mode = 'bl'
        elif near_x2 and near_y2:
            self._drag_mode = 'br'
        elif self.crop_x1 < rx < self.crop_x2 and self.crop_y1 < ry < self.crop_y2:
            self._drag_mode = 'move'
            self._drag_start = (rx, ry, self.crop_x1, self.crop_y1,
                                self.crop_x2, self.crop_y2)
        else:
            # Neues Rechteck ziehen
            self.crop_x1 = rx
            self.crop_y1 = ry
            self.crop_x2 = rx
            self.crop_y2 = ry
            self._drag_mode = 'br'

    def _on_motion(self, event):
        if not self._drag_mode:
            return
        rx, ry = self._canvas_to_rel(event.x, event.y)

        if self._drag_mode == 'move' and self._drag_start:
            srx, sry, sx1, sy1, sx2, sy2 = self._drag_start
            dx = rx - srx
            dy = ry - sry
            w = sx2 - sx1
            h = sy2 - sy1
            self.crop_x1 = max(0, min(1 - w, sx1 + dx))
            self.crop_y1 = max(0, min(1 - h, sy1 + dy))
            self.crop_x2 = self.crop_x1 + w
            self.crop_y2 = self.crop_y1 + h
        else:
            if 'l' in self._drag_mode:
                self.crop_x1 = min(rx, self.crop_x2 - 0.02)
            if 'r' in self._drag_mode:
                self.crop_x2 = max(rx, self.crop_x1 + 0.02)
            if 't' in self._drag_mode:
                self.crop_y1 = min(ry, self.crop_y2 - 0.02)
            if 'b' in self._drag_mode:
                self.crop_y2 = max(ry, self.crop_y1 + 0.02)

        self._draw()

    def _on_release(self, event):
        self._drag_mode = None
        self._drag_start = None
        # Normalisiere (x1 < x2, y1 < y2)
        if self.crop_x1 > self.crop_x2:
            self.crop_x1, self.crop_x2 = self.crop_x2, self.crop_x1
        if self.crop_y1 > self.crop_y2:
            self.crop_y1, self.crop_y2 = self.crop_y2, self.crop_y1
        self._draw()

    def _reset(self):
        self.crop_x1, self.crop_y1 = 0.0, 0.0
        self.crop_x2, self.crop_y2 = 1.0, 1.0
        self._draw()

    def _apply(self):
        """Wendet den Crop auf das Bild an."""
        x1 = int(self.crop_x1 * self.iw)
        y1 = int(self.crop_y1 * self.ih)
        x2 = int(self.crop_x2 * self.iw)
        y2 = int(self.crop_y2 * self.ih)

        if x2 - x1 < 10 or y2 - y1 < 10:
            messagebox.showwarning("Zu klein",
                "Der Ausschnitt ist zu klein.", parent=self.win)
            return

        self.app._push_undo("Zuschneiden")
        self.app.original_image = self.app.original_image[y1:y2, x1:x2].copy()

        mix = self.app._get_mix_matrix()
        from channel_swap import apply_to_image
        self.app.preview_image = apply_to_image(self.app.original_image, mix)
        self.app._reset_zoom()
        self.app._display_preview()
        self.app._update_histogram()
        self.app._update_status(f"Zugeschnitten: {x2-x1} x {y2-y1} px")
        self.win.destroy()
