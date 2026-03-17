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
    from lut_export import write_cube_lut, combined_lut
    HAS_LUT = True
except ImportError:
    HAS_LUT = False

from channel_swap import apply_to_image


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
            os.startfile(os.path.dirname(filepath))
