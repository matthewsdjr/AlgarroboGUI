"""Display, zoom, pan and navigate through multispectral images."""

import numpy as np
import cv2
from PIL import Image, ImageTk

from config.settings import Settings


# ── Display helpers ──────────────────────────────────────────────────────────

def mostrar_imagen(state, datos):
    """Render an image array into the main viewer label."""
    factor = Settings.DISPLAY_FACTOR
    a, b = state.a, state.b
    w = state.widgets

    if datos.dtype == np.uint8:
        visible = cv2.resize(
            datos, (int(b * (1 - factor)), int(a * (1 - factor))),
            interpolation=cv2.INTER_AREA,
        )
    elif datos.dtype == np.uint16:
        visible = (
            cv2.resize(
                datos, (int(b * (1 - factor)), int(a * (1 - factor))),
                interpolation=cv2.INTER_AREA,
            )
            / 255
        )
    else:
        visible = cv2.resize(
            datos, (int(b * (1 - factor)), int(a * (1 - factor))),
            interpolation=cv2.INTER_AREA,
        )

    im_mostrar = Image.fromarray(visible.astype("uint8") if visible.dtype != np.uint8 else visible)
    visibilito = ImageTk.PhotoImage(image=im_mostrar)
    w["ttk_label_imagen"].configure(image=visibilito)
    w["ttk_label_imagen"].image = visibilito


def mostrar_img_seg(state, img):
    """Render a segmented image into the segmentation panel."""
    w = state.widgets
    img_resized = cv2.resize(img, (int(1600 / 3), int(1300 / 3)), interpolation=None)
    im_mostrar = Image.fromarray(img_resized)
    visibilito = ImageTk.PhotoImage(image=im_mostrar)
    w["label_img_segm"].configure(image=visibilito)
    w["label_img_segm"].image = visibilito


def bandas_independientes(state, datos):
    """Display R, G, B band thumbnails in the left panel."""
    w = state.widgets
    labels = [w["label_capa_red"], w["label_capa_green"], w["label_capa_blue"]]
    thumb_w, thumb_h = int(1600 / 6), int(1300 / 6)

    for i in range(3):
        band_idx = 5 if (i == 2 and not getattr(state, "tiene_banda_azul", True)) else i + 1
        raw = cv2.resize(datos[band_idx], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        normalized = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        im_mostrar = Image.fromarray(normalized)
        visiblito = ImageTk.PhotoImage(image=im_mostrar)
        labels[i].configure(image=visiblito)
        labels[i].image = visiblito


# ── Navigation ───────────────────────────────────────────────────────────────

def siguiente(state):
    """Go to next band / image."""
    from tkinter import messagebox
    w = state.widgets
    state.contar_imagen += 1

    if state.contar_imagen == 6:
        state.contar_imagen = 0
        if np.all(state.puntos_imagen[:, :] != 0):
            messagebox.showinfo("Aviso", "Puntos generados correctamente")

    state.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
    mostrar_imagen(state, state.data_image[state.contar_imagen])
    w["lbl_img_selec"].configure(text=Settings.INDEX_LABELS[state.contar_imagen])
    state.escala = {"x": 1600, "y": 1300}
    _mostrar_banda(state)


def anterior(state):
    """Go to previous band / image."""
    w = state.widgets
    state.contar_imagen -= 1
    if state.contar_imagen <= 0:
        state.contar_imagen = 0

    state.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
    mostrar_imagen(state, state.data_image[state.contar_imagen])
    w["lbl_img_selec"].configure(text=Settings.INDEX_LABELS[state.contar_imagen])
    state.escala = {"x": 1600, "y": 1300}
    _mostrar_banda(state)


def siguiente_seg(state):
    """Navigate to the next SAM segmentation mask."""
    state.contar_mask += 1
    if state.contar_mask >= len(state.save_mask):
        state.contar_mask = 0
    mostrar_img_seg(state, state.save_mask[state.contar_mask]["mask"])
    mostrar_banda_seg(state)


def anterior_seg(state):
    """Navigate to the previous SAM segmentation mask."""
    if state.contar_mask <= 0:
        state.contar_mask = len(state.save_mask) - 1
    else:
        state.contar_mask -= 1
    mostrar_img_seg(state, state.save_mask[state.contar_mask]["mask"])
    mostrar_banda_seg(state)


def mostrar_banda_seg(state):
    w = state.widgets
    num = round(state.save_mask[state.contar_mask]["score"], 3)
    w["lbl_img_selec_seg"].configure(
        text=f"Mascara {state.contar_mask + 1} - Score: {num}"
    )


def _mostrar_banda(state):
    w = state.widgets
    w["lbl_img_selec"].configure(text=str(Settings.INDEX_LABELS[state.contar_imagen]))


# ── Zoom ─────────────────────────────────────────────────────────────────────

def zoom_in(state):
    a, b = state.a, state.b
    state.escala["x"] *= Settings.ZOOM_FACTOR_IN
    state.escala["y"] *= Settings.ZOOM_FACTOR_IN

    state.distancia_zoom["x"] = int((state.escala["x"] - b) / 2)
    state.distancia_zoom["y"] = int((state.escala["y"] - a) / 2)

    dz = state.distancia_zoom
    if dz["y"] < int(a / 2) and dz["x"] < int(b / 2):
        src = state.data_image[-1] if len(state.data_image) == 7 else state.data_image[state.contar_imagen]
        data = src[dz["y"]: a - dz["y"], dz["x"]: b - dz["x"]]
        state.limites["x_1"] = dz["x"]
        state.limites["x_2"] = b - dz["x"]
        state.limites["y_1"] = dz["y"]
        state.limites["y_2"] = a - dz["y"]
        mostrar_imagen(state, data)
    else:
        state.escala["x"] /= Settings.ZOOM_FACTOR_IN
        state.escala["y"] /= Settings.ZOOM_FACTOR_IN


def zoom_out(state):
    a, b = state.a, state.b
    state.escala["x"] *= Settings.ZOOM_FACTOR_OUT
    state.escala["y"] *= Settings.ZOOM_FACTOR_OUT

    state.distancia_zoom["x"] = int((state.escala["x"] - b) / 2)
    state.distancia_zoom["y"] = int((state.escala["y"] - a) / 2)

    dz = state.distancia_zoom
    if dz["y"] > 0 or dz["x"] > 0:
        src = state.data_image[-1] if len(state.data_image) == 7 else state.data_image[state.contar_imagen]
        data = src[dz["y"]: a - dz["y"], dz["x"]: b - dz["x"]]
        state.limites["x_1"] = dz["x"]
        state.limites["x_2"] = b - dz["x"]
        state.limites["y_1"] = dz["y"]
        state.limites["y_2"] = a - dz["y"]
        mostrar_imagen(state, data)
    else:
        src = state.data_image[-1] if len(state.data_image) == 7 else state.data_image[state.contar_imagen]
        mostrar_imagen(state, src)
        state.escala = {"x": 1600, "y": 1300}
        state.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}


def zoom_scroll(state, event):
    if event.delta > 0:
        zoom_in(state)
    elif event.delta < 0:
        zoom_out(state)


# ── Mouse interaction ────────────────────────────────────────────────────────

def mover_mouse(state, event):
    a, b = state.a, state.b
    dz = state.distancia_zoom
    dm = state.distancia_move

    if state.presionado and not state.flag_roi and not state.flag_sam:
        margen = {
            "y_1": dz["y"] - dm["y"],
            "y_2": a - dz["y"] - dm["y"],
            "x_1": dz["x"] - dm["x"],
            "x_2": b - dz["x"] - dm["x"],
        }

        dm["x"] = event.x - state.drag_data["x"]
        dm["y"] = event.y - state.drag_data["y"]

        src = state.data_image[-1] if len(state.data_image) == 7 else state.data_image[state.contar_imagen]
        data = src[margen["y_1"]: margen["y_2"], margen["x_1"]: margen["x_2"]]

        if margen["y_1"] >= 0 and margen["y_2"] <= a and margen["x_1"] >= 0 and margen["x_2"] <= b:
            mostrar_imagen(state, data)
            state.limites["x_1"] = margen["x_1"]
            state.limites["x_2"] = b - margen["x_2"]
            state.limites["y_1"] = margen["y_1"]
            state.limites["y_2"] = a - margen["y_2"]

    elif state.presionado and state.flag_roi and not state.flag_sam:
        lim = state.limites
        origenx = event.x
        origeny = event.y

        state.trazo["x"] = int((1600 - lim["x_2"] - lim["x_1"]) * origenx / 800 + lim["x_1"])
        state.trazo["y"] = int((1300 - lim["y_2"] - lim["y_1"]) * origeny / 650 + lim["y_1"])
        state.contorno.append([state.trazo["x"], state.trazo["y"]])
        tx, ty = state.trazo["x"], state.trazo["y"]
        state.puntos_visibles[ty - 2: ty + 2, tx - 2: tx + 2, :] = (255, 0, 0)
        mostrar_imagen(state, state.puntos_visibles)


def mascara_presionado(state, event):
    """Handle left-click press: start drag or collect SAM points."""
    import tkinter as tk
    from tkinter import messagebox

    state.presionado = True
    state.drag_data["x"] = event.x
    state.drag_data["y"] = event.y

    if not state.data_image:
        messagebox.showinfo("Aviso", "Adjuntar imagen")
    else:
        state.puntos_visibles = state.data_image[0].copy()

    if state.flag_sam:
        _collect_sam_point(state, event)


def _collect_sam_point(state, event):
    from tkinter import messagebox
    import numpy as np

    lim = state.limites
    a, b = state.a, state.b
    c_x, c_y = event.x, event.y

    state.point_sam["x"] = int((1600 - lim["x_2"] - lim["x_1"]) * c_x / 800 + lim["x_1"])
    state.point_sam["y"] = int((1300 - lim["y_2"] - lim["y_1"]) * c_y / 650 + lim["y_1"])

    state.puntos_sam.append([state.point_sam["x"], state.point_sam["y"]])

    if state.input_label is None:
        state.input_label = np.array([1])
    else:
        state.input_label = np.append(state.input_label, 1)

    if len(state.puntos_sam) > 1:
        ans = messagebox.askquestion("Aviso", "¿Desea agregar otro punto?")
        if ans == "no":
            state.puntos_sam.pop(-1)
            state.input_label = np.delete(state.input_label, -1)
            return

    img_cir_sam = state.data_image[state.contar_imagen].copy()
    for punto in state.puntos_sam:
        img_cir_sam = cv2.circle(img_cir_sam, (int(punto[0]), int(punto[1])), 10, (0, 255, 0), -1)
    img_pto_sam = img_cir_sam[lim["y_1"]: a - lim["y_2"], lim["x_1"]: b - lim["x_2"]]

    mostrar_imagen(state, img_pto_sam)
    state.input_point = np.array(state.puntos_sam)
