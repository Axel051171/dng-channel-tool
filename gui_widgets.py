"""
Custom Widgets für DNG Channel Tool

Enthält wiederverwendbare UI-Komponenten:
- AutocompleteCombobox: Combobox mit Suchfilter beim Tippen
- HistogramWidget: RGB-Histogramm-Anzeige
- ToolTip: Tooltip-Anzeige bei Hover über Widgets
- ProgressDialog: Fortschrittsanzeige für längere Operationen
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
import numpy as np


class AutocompleteCombobox(ttk.Combobox):
    """Combobox mit Suchfilter beim Tippen."""

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
    """RGB-Histogramm-Widget mit Clipping-Warnung."""

    def __init__(self, master, height=100, **kwargs):
        super().__init__(master, height=height, bg='#1a1a1a', highlightthickness=0, **kwargs)
        self._height = height
        self._clipping_info = ""

    def update_histogram(self, image: Optional[np.ndarray]):
        self.delete('all')
        if image is None:
            return

        w = self.winfo_width()
        if w < 10:
            w = 300

        h = self._height
        colors = ['#ff3333', '#33cc33', '#3366ff']

        total_pixels = image.shape[0] * image.shape[1]
        clip_shadows = 0
        clip_highlights = 0

        for ch, color in enumerate(colors):
            channel = image[:, :, ch].ravel()
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            if hist.max() == 0:
                continue

            # Clipping zählen
            clip_shadows += int(hist[0])
            clip_highlights += int(hist[255])

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

            # Polygon ohne Füllung. `stipple` darf laut Tk-Doku nicht leer
            # sein – nur setzen wenn er gebraucht wird (Kanal != Rot).
            poly_kwargs = {'fill': '', 'outline': color, 'width': 1}
            if ch > 0:
                poly_kwargs['stipple'] = 'gray50'
            self.create_polygon(coords, **poly_kwargs)

        # Clipping-Indikatoren zeichnen
        shadow_pct = clip_shadows / (total_pixels * 3) * 100
        highlight_pct = clip_highlights / (total_pixels * 3) * 100

        if shadow_pct > 0.5:
            # Blaues Dreieck links = Schatten-Clipping
            self.create_polygon(2, h - 2, 2, h - 14, 10, h - 2,
                                fill='#4488ff', outline='')
            self.create_text(14, h - 8, text=f"{shadow_pct:.1f}%",
                             fill='#4488ff', anchor='w', font=('Consolas', 7))

        if highlight_pct > 0.5:
            # Rotes Dreieck rechts = Lichter-Clipping
            self.create_polygon(w - 2, h - 2, w - 2, h - 14, w - 10, h - 2,
                                fill='#ff4444', outline='')
            self.create_text(w - 14, h - 8, text=f"{highlight_pct:.1f}%",
                             fill='#ff4444', anchor='e', font=('Consolas', 7))

        self._clipping_info = (f"Schatten: {shadow_pct:.1f}%  "
                                f"Lichter: {highlight_pct:.1f}%")


class ToolTip:
    """Tooltip-Anzeige bei Hover über ein Widget.

    Verwendung:
        ToolTip(button, "Das ist ein Hilfetext")
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None

        widget.bind('<Enter>', self._on_enter, add='+')
        widget.bind('<Leave>', self._on_leave, add='+')
        widget.bind('<ButtonPress>', self._on_leave, add='+')

    def _on_enter(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self._tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.wm_attributes('-topmost', True)

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background='#ffffe1', foreground='#333333',
                         relief=tk.SOLID, borderwidth=1,
                         font=('Segoe UI', 9), padx=6, pady=3,
                         wraplength=300)
        label.pack()

    def _hide(self):
        tw = self._tipwindow
        self._tipwindow = None
        if tw:
            tw.destroy()

    def update_text(self, text: str):
        """Aktualisiert den Tooltip-Text."""
        self.text = text


class ProgressDialog:
    """Modaler Fortschrittsdialog für längere Operationen.

    Verwendung:
        dlg = ProgressDialog(parent, "Export läuft…", maximum=100)
        dlg.update(50, "Datei 5 von 10…")
        dlg.close()
    """

    def __init__(self, parent: tk.Widget, title: str = "Bitte warten…",
                 maximum: float = 100, cancelable: bool = False,
                 on_cancel: Optional[Callable] = None):
        self.cancelled = False
        self._on_cancel = on_cancel

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("420x140")
        self.win.transient(parent)
        self.win.grab_set()
        self.win.resizable(False, False)
        self.win.protocol('WM_DELETE_WINDOW', self._do_cancel if cancelable else lambda: None)

        frame = ttk.Frame(self.win, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Starte…")
        self.status_label = ttk.Label(frame, textvariable=self.status_var,
                                       wraplength=380)
        self.status_label.pack(anchor='w', pady=(0, 8))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(frame, variable=self.progress_var,
                                         maximum=maximum, length=380)
        self.progress.pack(fill=tk.X)

        self.detail_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.detail_var,
                  foreground='gray').pack(anchor='w', pady=(4, 0))

        if cancelable:
            ttk.Button(frame, text="Abbrechen",
                       command=self._do_cancel).pack(anchor='e', pady=(5, 0))

    def update(self, value: float, status: str = "", detail: str = ""):
        """Aktualisiert Fortschritt und Statustext.

        Tolerant gegenüber Aufrufen nach `close()`: dann wird stillschweigend
        ignoriert (kommt bei Worker-Threads vor, die einen letzten Update-Schub
        schicken nachdem der Dialog bereits zu ist).
        """
        try:
            if not self.win.winfo_exists():
                return
            self.progress_var.set(value)
            if status:
                self.status_var.set(status)
            if detail:
                self.detail_var.set(detail)
            self.win.update_idletasks()
        except tk.TclError:
            pass

    def close(self):
        """Schließt den Dialog."""
        try:
            self.win.grab_release()
            self.win.destroy()
        except tk.TclError:
            pass

    def _do_cancel(self):
        self.cancelled = True
        if self._on_cancel:
            self._on_cancel()
        self.close()
