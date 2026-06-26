"""
Global application state shared between all UI modules.
Replaces scattered global variables with a single managed object.
"""

import numpy as np


class AppState:
    """Holds mutable runtime state that multiple modules need to access."""

    def __init__(self):
        # ── Image data ──────────────────────────────────────────
        self.data_image: list = []
        self.file_path: str = ""
        self.a: int = 0  # image height
        self.b: int = 0  # image width

        # ── Navigation ──────────────────────────────────────────
        self.contar_imagen: int = 0
        # Band indices actually delivered by the drone for this capture
        # (0=RGB, 1=R, 2=G, 3=B, 4=RE, 5=NIR). Blue (3) is dropped when the
        # DJI capture has no MS_B.TIF, so it is never shown in the viewer.
        self.available_bands: list = [0, 1, 2, 3, 4, 5]
        self.escala = {"x": 0, "y": 0}
        self.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
        self.distancia_zoom = {"x": 0, "y": 0}
        self.distancia_move = {"x": 0, "y": 0}
        self.drag_data = {"x": 0, "y": 0}
        self.trazo = {"x": 0, "y": 0}

        # ── Flags ───────────────────────────────────────────────
        self.flag_correccion: int = 0
        self.presionado: bool = False
        self.flag_roi: bool = False
        self.flag_trazo: bool = False
        self.flag_sam: bool = False
        self.correction_done: bool = False
        self.tiene_banda_azul: bool = True  # False when DJI format has no MS_B.TIF

        # ── Segmentation ────────────────────────────────────────
        self.contorno: list = []
        self.puntos_visibles = None
        self.img_act_label = None
        self.img_segmenetada = None
        self.imagen_segementada = None
        self.mask = None

        # ── SAM segmentation ────────────────────────────────────
        self.puntos_sam: list = []
        self.point_sam = {"x": 0, "y": 0}
        self.cont_puntos_sam: int = 0
        self.input_point = None
        self.input_label = None
        self.save_mask: list = []
        self.contar_mask: int = 2
        self.fin_segm = None

        # ── Indices ─────────────────────────────────────────────
        self.Indices = None
        self.puntos_imagen = None
        # Single-index colormap view (full-resolution float values + metadata
        # used for the per-pixel cursor readout and the colorbar legend).
        self.index_vals = None
        self.index_name = None
        self.index_def = None

        # ── Correction ──────────────────────────────────────────
        self.imagenes_corregidas: list = []

        # ── Coordinates ─────────────────────────────────────────
        self.latitude = None
        self.longitude = None

        # ── Models (loaded at runtime) ──────────────────────────
        self.sam = None
        self.net = None           # AlexNet ONNX
        self.AlgarroboNet = None  # AlgarroboNet ONNX

        # ── Widget references (set by UI builders) ──────────────
        self.widgets = {}

    def reset_navigation(self):
        """Reset zoom / pan state to defaults."""
        self.escala = {"x": 1600, "y": 1300}
        self.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
        self.distancia_zoom = {"x": 0, "y": 0}
        self.distancia_move = {"x": 0, "y": 0}
