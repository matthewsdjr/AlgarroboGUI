"""Manual ROI-based segmentation using freehand drawing."""

import numpy as np
import cv2
from tkinter import messagebox
from PIL import Image, ImageTk

from src.image_processing.image_viewer import mostrar_imagen


def roi(state):
    """Toggle the ROI drawing (pencil) tool."""
    w = state.widgets
    state.flag_roi = not state.flag_roi
    if state.flag_roi:
        w["boton_lapiz"].configure(fg_color="red")
    else:
        w["boton_lapiz"].configure(fg_color="Teal")
        state.flag_roi = False


def mascara(state):
    """Create a segmentation mask from the drawn contour."""
    a, b = state.a, state.b
    w = state.widgets
    display_img = state.data_image[0].copy()

    if state.flag_roi:
        state.mask = np.zeros((a, b), np.uint8)
        arr = np.array(state.contorno)
        cv2.fillPoly(state.mask, pts=[arr], color=255)

        state.imagen_segementada = cv2.bitwise_and(display_img, display_img, mask=state.mask)

        seg_resize = cv2.resize(
            state.imagen_segementada, (int(1600 / 3), int(1300 / 3)),
            interpolation=cv2.INTER_AREA,
        )

        img_seg = Image.fromarray(seg_resize)
        img_seg_tk = ImageTk.PhotoImage(image=img_seg)

        w["label_img_segm"].configure(image=img_seg_tk)
        w["label_img_segm"].image = img_seg_tk
        w["lbl_img_selec_seg"].configure(text="Segmentación Manual")
        messagebox.showinfo("Aviso", "Segmentación correcta")
    else:
        messagebox.showerror("Error", "Realice el trazo al objeto a segmentar")
