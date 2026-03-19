"""Calculate vegetation indices from multispectral bands."""

import numpy as np
import cv2
from PIL import Image, ImageTk

from config.settings import Settings


def calcIndices(state, data):
    """Compute 24 spectral indices and store in state.Indices."""
    R = data[1] / (2 ** 16)
    G = data[2] / (2 ** 16)
    B = data[3] / (2 ** 16)
    ER = data[4] / (2 ** 16)
    NIR = data[5] / (2 ** 16)

    ExG = (2 * G - R - B - (-2)) / (2 - (-2))
    ExGR = ((2 * G - R - B) - (1.44 * R - G) - (-3.44)) / (3 - (-3.44))
    NGRDI = (((G - R) / (G + R)) - (-1)) / (1 - (-1))
    NGBDI = (((G - B) / (G + B)) - (-1)) / (1 - (-1))
    RGRI = R / G
    GBRI = B / G
    CIVE = ((0.441 * R - 0.811 * G + 0.385 * B + 18.78745) - 17.8) / (20 - 17.8)
    a_coeff = 0.667
    VEG = (G / (R ** a_coeff) * (B ** (1 - a_coeff))) / 20
    RGBVI = (((G ** 2) - (B * (R ** 2))) - (-1)) / (1 - (-1))
    MGRVI = ((((G ** 2) - (R ** 2)) / ((G ** 2) + (R ** 2))) - (-1)) / (1 - (-1))

    NDVI = (((NIR - R) / (NIR + R)) - (-1)) / (1 - (-1))
    RVI = (NIR / R - 0) / (1 - 0)
    DVI = ((NIR - R) - (-1)) / (1 - (-1))
    epsilon = 1e-6
    EVI = ((((NIR - R) * 2.5) / (NIR + (R * 6) - (B * 7.5) + 1 + epsilon)) - (-2.5)) / (2.5 - (-2.5))
    REVI = (((NIR / ER) - 1) - (-1)) / (20 - (-1))
    NDRE = (((NIR - ER) / (NIR + ER)) - (-1)) / (1 - (-1))
    RERVI = NIR / ER
    REDVI = ((NIR - ER) - (-1)) / (1 - (-1))
    SCCI = NDRE / NDVI

    state.Indices = np.stack(
        [R, G, B, ER, NIR, ExG, ExGR, NGRDI, NGBDI, RGRI, GBRI,
         CIVE, VEG, RGBVI, MGRVI, NDVI, RVI, DVI, EVI, REVI, NDRE,
         RERVI, REDVI, SCCI],
        axis=2,
    )
    state.Indices[np.isnan(state.Indices)] = 0
    state.Indices[np.isinf(state.Indices)] = 1


def boton_indices(state):
    """Read combo-box selections and plot the combined index image."""
    import tkinter as tk
    w = state.widgets

    primer_canal = w["cmb_1"].current() - 1
    segundo_canal = w["cmb_2"].current() - 1
    tercer_canal = w["cmb_3"].current() - 1

    if -1 in (primer_canal, segundo_canal, tercer_canal):
        tk.messagebox.showinfo("ALERTA", "DEBE SELECCIONAR UN ITEM DE CADA COMBOBOX")
    else:
        graficar_indices(state, state.Indices, primer_canal, segundo_canal, tercer_canal)


def boton_indices_event(state, event):
    boton_indices(state)


def graficar_indices(state, ind, p, s, t):
    """Merge 3 selected index channels into an RGB display."""
    from src.image_processing.image_viewer import mostrar_imagen
    w = state.widgets

    combined = cv2.merge([ind[:, :, p], ind[:, :, s], ind[:, :, t]])
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    combined_final = combined.astype(np.uint8)

    state.img_act_label = combined_final

    if len(state.data_image) >= 7:
        state.data_image.pop(-1)
    state.data_image.append(state.img_act_label)

    rgb_completo = cv2.resize(combined_final, (800, 650), interpolation=cv2.INTER_AREA)
    im_bg = Image.fromarray(rgb_completo)
    photo = ImageTk.PhotoImage(image=im_bg)
    w["ttk_label_imagen"].configure(image=photo)
    w["ttk_label_imagen"].image = photo
    w["lbl_img_selec"].configure(
        text=f"{w['cmb_1'].get()} - {w['cmb_2'].get()} - {w['cmb_3'].get()}"
    )

    _update_band_thumbnail(w["label_capa_red"], ind, p)
    _update_band_thumbnail(w["label_capa_green"], ind, s)
    _update_band_thumbnail(w["label_capa_blue"], ind, t)


def _update_band_thumbnail(label, ind, channel):
    image = cv2.merge([ind[:, :, channel]])
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = cv2.resize(image, (int(1600 / 6), int(1300 / 6)), interpolation=cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
    label.configure(image=photo)
    label.image = photo
