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
import tkinter as tk

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
