"""
Undo/Redo System für DNG Channel Tool

Modell:
  - push(state) speichert den AKTUELLEN Zustand (BEVOR eine neue Aktion läuft).
  - undo(current) entfernt den Top-Snapshot, schiebt `current` (=Zustand
    NACH der letzten Aktion) auf den Redo-Stack und gibt den entfernten
    Snapshot zurück — also den Zustand VOR der Aktion.
  - redo(current) ist symmetrisch.

Die Description eines Snapshots beschreibt die Aktion, die NACH ihm
ausgeführt wurde (Adobe-Lightroom-Konvention). Beim Undo wird genau diese
Description als "Rückgängig: <Aktion>" angezeigt.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)

MAX_UNDO_STEPS = 20


@dataclass
class UndoState:
    """Ein gespeicherter Zustand."""
    preview_image: np.ndarray
    mix_matrix: np.ndarray  # 3x3
    description: str = ""


class UndoManager:
    """
    Verwaltet Undo/Redo für Bildverarbeitungs-Operationen.

    Speichert bis zu MAX_UNDO_STEPS Zustände. Beim Hinzufügen eines
    neuen Zustands wird der Redo-Stack gelöscht.
    """

    def __init__(self, max_steps: int = MAX_UNDO_STEPS):
        self._undo_stack: List[UndoState] = []
        self._redo_stack: List[UndoState] = []
        self._max_steps = max_steps
        self._callbacks: List[Callable] = []

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        """Beschreibung der Aktion, die rückgängig gemacht wird."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    @property
    def redo_description(self) -> str:
        """Beschreibung der Aktion, die wiederhergestellt wird."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""

    def push(self, preview_image: np.ndarray, mix_matrix: np.ndarray,
             description: str = ""):
        """
        Speichert den aktuellen Zustand auf dem Undo-Stack.

        Args:
            preview_image: Aktuelles Vorschaubild (wird kopiert)
            mix_matrix: Aktuelle 3x3 Mix-Matrix (wird kopiert)
            description: Beschreibung der FOLGENDEN Aktion
        """
        state = UndoState(
            preview_image=preview_image.copy(),
            mix_matrix=mix_matrix.copy(),
            description=description,
        )
        self._undo_stack.append(state)
        # Neue Aktion bricht die Redo-Kette
        self._redo_stack.clear()

        while len(self._undo_stack) > self._max_steps:
            self._undo_stack.pop(0)

        logger.debug("Undo: Push '%s' (Stack: %d)", description, len(self._undo_stack))
        self._notify()

    def undo(self, current_preview: Optional[np.ndarray] = None,
             current_mix: Optional[np.ndarray] = None) -> Optional[UndoState]:
        """
        Macht die letzte Aktion rückgängig.

        Args:
            current_preview: Aktuelles Vorschaubild (Zustand NACH der Aktion).
                Wird auf den Redo-Stack gepackt, damit Redo den korrekten
                Zustand wiederherstellen kann. Wenn None, kann nach diesem
                Undo kein Redo erfolgen.
            current_mix: Aktuelle Mix-Matrix (3x3).

        Returns:
            UndoState mit dem Zustand VOR der Aktion und der Description
            der Aktion (zur Anzeige "Rückgängig: <X>"). None bei leerem Stack.
        """
        if not self._undo_stack:
            logger.debug("Undo: Stack leer")
            return None

        before = self._undo_stack.pop()

        if current_preview is not None and current_mix is not None:
            self._redo_stack.append(UndoState(
                preview_image=np.ascontiguousarray(current_preview).copy(),
                mix_matrix=np.asarray(current_mix).copy(),
                description=before.description,
            ))
        else:
            self._redo_stack.clear()

        logger.debug("Undo: '%s' (Stack: %d, Redo: %d)",
                     before.description, len(self._undo_stack), len(self._redo_stack))
        self._notify()
        return before

    def redo(self, current_preview: Optional[np.ndarray] = None,
             current_mix: Optional[np.ndarray] = None) -> Optional[UndoState]:
        """
        Stellt eine zurückgenommene Aktion wieder her.

        Args:
            current_preview: Aktuelles Vorschaubild (vor Redo). Wird auf
                den Undo-Stack zurückgepackt. None deaktiviert Re-Undo.
            current_mix: Aktuelle Mix-Matrix.

        Returns:
            UndoState mit dem wiederhergestellten Zustand. None bei leerem Stack.
        """
        if not self._redo_stack:
            logger.debug("Redo: Stack leer")
            return None

        after = self._redo_stack.pop()

        if current_preview is not None and current_mix is not None:
            self._undo_stack.append(UndoState(
                preview_image=np.ascontiguousarray(current_preview).copy(),
                mix_matrix=np.asarray(current_mix).copy(),
                description=after.description,
            ))
            while len(self._undo_stack) > self._max_steps:
                self._undo_stack.pop(0)

        logger.debug("Redo: '%s' (Stack: %d, Redo: %d)",
                     after.description, len(self._undo_stack), len(self._redo_stack))
        self._notify()
        return after

    def clear(self):
        """Löscht beide Stacks (z.B. beim Laden eines neuen Bildes)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.debug("Undo: Stacks gelöscht")
        self._notify()

    def on_change(self, callback: Callable):
        """Registriert einen Callback der bei Stack-Änderungen aufgerufen wird."""
        self._callbacks.append(callback)

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning("Undo-Callback fehlgeschlagen: %s", e)
