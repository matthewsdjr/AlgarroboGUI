"""SAM (Segment Anything Model) based segmentation."""

import numpy as np
from tkinter import messagebox
from segment_anything import SamPredictor

from src.image_processing.image_viewer import mostrar_imagen, mostrar_mascara_sam


def sam_segm_flag(state):
    """Activate SAM point-collection mode (clears any previous points/masks)."""
    if not state.data_image:
        messagebox.showinfo("SAM", "Adjunte una imagen primero.")
        return
    state.save_mask = []
    state.puntos_sam = []
    state.input_label = None
    state.contar_mask = 0
    state.flag_sam = True
    mostrar_imagen(state, state.data_image[state.contar_imagen])
    state.widgets["lbl_img_selec"].configure(
        text="SAM · marca uno o varios puntos sobre el ROI")


def sam_segm(state):
    """Run SAM and display the 3 candidate masks in the viewer for selection."""
    if state.sam is None:
        messagebox.showwarning("SAM", "El modelo SAM aún no está disponible.")
        return
    if not state.puntos_sam:
        messagebox.showinfo("SAM", "Primero marca al menos un punto sobre el ROI.")
        return

    predictor = SamPredictor(state.sam)
    predictor.set_image(state.data_image[0])
    display_img = state.data_image[0].copy()

    masks, scores, _ = predictor.predict(
        point_coords=state.input_point,
        point_labels=state.input_label,
        multimask_output=True,
    )

    state.save_mask = []
    for mask, score in zip(masks, scores):
        fondo_negro = np.zeros_like(display_img)
        imagen_segmentada = np.where(mask[:, :, None], display_img, fondo_negro)
        state.save_mask.append({"mask": imagen_segmentada, "score": float(score)})

    state.contar_mask = 0
    state.flag_sam = False
    mostrar_mascara_sam(state)  # shows mask 1/3 in the main viewer
