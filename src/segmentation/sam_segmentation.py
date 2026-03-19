"""SAM (Segment Anything Model) based segmentation."""

import numpy as np
from tkinter import messagebox
from segment_anything import SamPredictor

from src.image_processing.image_viewer import mostrar_imagen, mostrar_img_seg, mostrar_banda_seg


def sam_segm_flag(state):
    """Activate SAM point collection mode."""
    state.save_mask = []
    state.puntos_sam = []
    state.input_label = None
    mostrar_imagen(state, state.data_image[state.contar_imagen])
    state.flag_sam = True


def sam_segm(state):
    """Run SAM prediction using collected points."""
    predictor = SamPredictor(state.sam)
    predictor.set_image(state.data_image[0])

    display_img = state.data_image[0].copy()

    masks, scores, _ = predictor.predict(
        point_coords=state.input_point,
        point_labels=state.input_label,
        multimask_output=True,
    )

    for mask, score in zip(masks, scores):
        fondo_negro = np.zeros_like(display_img)
        imagen_segmentada = np.where(mask[:, :, None], display_img, fondo_negro)
        state.save_mask.append({
            "mask": imagen_segmentada,
            "score": score,
        })

    mostrar_img_seg(state, state.save_mask[state.contar_mask]["mask"])
    mostrar_banda_seg(state)
    messagebox.showinfo("Aviso", "Segmentación correcta")
    state.flag_sam = False
