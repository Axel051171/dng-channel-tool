"""
Custom Widgets für DNG Channel Tool

Enthält wiederverwendbare UI-Komponenten:
- AutocompleteCombobox: Combobox mit Suchfilter beim Tippen
- HistogramWidget: RGB-Histogramm-Anzeige
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
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
    """RGB-Histogramm-Widget."""

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
