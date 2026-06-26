"""
AlgarroboInterfaz — Entry point
================================
Desktop application for multispectral drone image analysis,
SAM-based tree segmentation and CNN classification of
Prosopis pallida (Algarrobo) trees.

Run:  python main.py
"""

import sys
import os
import ctypes
import tkinter as tk

# Declare Per-Monitor DPI awareness before Tk is created so Windows does not
# bitmap-scale the window (which blurs the title bar and all text).
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from src.app_state import AppState
from src.ui.login_window import build_login_window
from src.ui.main_window import build_main_window
from src.classification.model_loader import iniciar_carga_modelo


def main():
    Settings.ensure_dirs()

    state = AppState()

    root = tk.Tk()

    def on_login_success():
        build_main_window(root, state)
        iniciar_carga_modelo(state)

    build_login_window(root, on_login_success)

    root.mainloop()


if __name__ == "__main__":
    main()
