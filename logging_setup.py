"""
Logging-Konfiguration für DNG Channel Tool

Konfiguriert ein einheitliches Logging-System für alle Module.
Log-Ausgabe geht an die Konsole (INFO) und optional an eine Datei (DEBUG).
"""

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Konfiguriert das Logging-System.

    Args:
        level: Log-Level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optionaler Pfad zur Log-Datei

    Returns:
        Root-Logger für das Projekt
    """
    root_logger = logging.getLogger("dng_channel_tool")

    # Nicht doppelt konfigurieren
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Konsolen-Handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Datei-Handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        root_logger.addHandler(file_handler)

    return root_logger
