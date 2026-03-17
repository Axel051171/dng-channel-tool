"""
Undo/Redo System für DNG Channel Tool

Verwaltet einen Stack von Bildverarbeitungs-Zuständen.
Jeder Zustand speichert das Vorschaubild und die aktive Mix-Matrix.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
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

    Speichert bis zu MAX_UNDO_STEPS Zustände.
    Beim Hinzufügen eines neuen Zustands wird der Redo-Stack gelöscht.
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
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    @property
    def redo_description(self) -> str:
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""

    def push(self, preview_image: np.ndarray, mix_matrix: np.ndarray,
             description: str = ""):
        """
        Speichert den aktuellen Zustand auf dem Undo-Stack.

        Args:
            preview_image: Aktuelles Vorschaubild
            mix_matrix: Aktuelle 3x3 Mix-Matrix
            description: Beschreibung der Aktion (z.B. "R↔B Tausch")
        """
        state = UndoState(
            preview_image=preview_image.copy(),
            mix_matrix=mix_matrix.copy(),
            description=description,
        )
        self._undo_stack.append(state)

        # Redo-Stack leeren (neue Aktion bricht die Redo-Kette)
        self._redo_stack.clear()

        # Stack-Größe begrenzen
        while len(self._undo_stack) > self._max_steps:
            self._undo_stack.pop(0)

        logger.debug("Undo: Push '%s' (Stack: %d)", description, len(self._undo_stack))
        self._notify()

    def undo(self) -> Optional[UndoState]:
        """
        Stellt den vorherigen Zustand wieder her.

        Returns:
            Der vorherige Zustand oder None wenn der Stack leer ist.
        """
        if not self._undo_stack:
            logger.debug("Undo: Stack leer")
            return None

        state = self._undo_stack.pop()
        self._redo_stack.append(state)
        logger.debug("Undo: '%s' (Stack: %d, Redo: %d)",
                     state.description, len(self._undo_stack), len(self._redo_stack))
        self._notify()

        # Gib den Zustand VOR dieser Aktion zurück
        if self._undo_stack:
            return self._undo_stack[-1]
        return state

    def redo(self) -> Optional[UndoState]:
        """
        Stellt den nächsten Zustand wieder her (nach einem Undo).

        Returns:
            Der wiederhergestellte Zustand oder None.
        """
        if not self._redo_stack:
            logger.debug("Redo: Stack leer")
            return None

        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        logger.debug("Redo: '%s' (Stack: %d, Redo: %d)",
                     state.description, len(self._undo_stack), len(self._redo_stack))
        self._notify()
        return state

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
            except Exception:
                pass
