"""Manual ROI-based segmentation using freehand drawing."""

import numpy as np
import cv2
from tkinter import messagebox
from PIL import Image, ImageTk

from src.image_processing.image_viewer import mostrar_imagen


def roi(state):
    """Toggle the ROI drawing tool (kept for backward compatibility).

    The floating pencil button was removed from the viewer; manual segmentation
    is now activated from the Segmentación tool panel.
    """
    state.flag_roi = not state.flag_roi
    btn = state.widgets.get("boton_lapiz")
    if btn is not None:
        btn.configure(fg_color="red" if state.flag_roi else "Teal")


def mascara(state):
    """Create a segmentation mask from the drawn contour and show it in the viewer."""
    a, b = state.a, state.b
    w = state.widgets

    if not state.flag_roi or len(state.contorno) < 3:
        messagebox.showerror("Error", "Activa el lápiz y traza el contorno del objeto a segmentar.")
        return

    display_img = state.data_image[0].copy()
    state.mask = np.zeros((a, b), np.uint8)
    arr = np.array(state.contorno)
    cv2.fillPoly(state.mask, pts=[arr], color=255)

    state.imagen_segementada = cv2.bitwise_and(display_img, display_img, mask=state.mask)
    # Use the manual ROI for classification (discard any previous SAM masks).
    state.save_mask = []
    state.flag_roi = False  # finished drawing

    mostrar_imagen(state, state.imagen_segementada)
    w["lbl_img_selec"].configure(text="Segmentación manual")
    lbl_seg = w.get("lbl_img_selec_seg")
    if lbl_seg is not None:
        lbl_seg.configure(text="ROI manual listo para clasificar")
