"""Geometric correction of multispectral band alignment."""

import numpy as np
import cv2
from tkinter import messagebox

from src.image_processing.image_viewer import mostrar_imagen, bandas_independientes
from src.image_processing.spectral_indices import calcIndices


def punto(state, event):
    """Record a correction reference point on right-click."""
    if state.flag_correccion != 1:
        return

    a, b = state.a, state.b
    lim = state.limites

    point = {
        "x": int((1600 - lim["x_2"] - lim["x_1"]) * event.x / 800 + lim["x_1"]),
        "y": int((1300 - lim["y_2"] - lim["y_1"]) * event.y / 650 + lim["y_1"]),
    }

    state.puntos_imagen[state.contar_imagen, :] = [point["x"], point["y"]]

    img_cir = cv2.circle(
        state.data_image[state.contar_imagen].copy(),
        (int(state.puntos_imagen[state.contar_imagen, 0]),
         int(state.puntos_imagen[state.contar_imagen, 1])),
        5, (0, 0, 255), -1,
    )
    img_pto = img_cir[lim["y_1"]: a - lim["y_2"], lim["x_1"]: b - lim["x_2"]]
    mostrar_imagen(state, img_pto)


def correccion_imagen(state):
    """Apply affine translation to align all bands to band 0."""
    a, b = state.a, state.b
    distancia = np.zeros((1, 2))
    state.imagenes_corregidas = []

    for i in range(6):
        distancia[0, 0] = state.puntos_imagen[0, 0] - state.puntos_imagen[i, 0]
        distancia[0, 1] = state.puntos_imagen[0, 1] - state.puntos_imagen[i, 1]
        M = np.float32([[1, 0, distancia[0, 0]], [0, 1, distancia[0, 1]]])
        image_out = cv2.warpAffine(state.data_image[i], M, (b, a))
        state.imagenes_corregidas.append(image_out)

    bandas_independientes(state, state.imagenes_corregidas)
    mostrar_imagen(state, state.imagenes_corregidas[0])
    messagebox.showinfo("Corrección Geométrica", "Imágenes corregidas con éxito")

    if state.flag_correccion == 0:
        calcIndices(state, state.data_image)
    else:
        calcIndices(state, state.imagenes_corregidas)
