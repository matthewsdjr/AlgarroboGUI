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


def bandas_independientes(state, datos):
    """Display R, G, B band thumbnails on the legacy (now hidden) band labels.

    Kept for backward compatibility with the geometric-correction routine; the
    single-viewer layout no longer shows a left thumbnail panel, so the target
    labels may be absent. Silently skip when they are.
    """
    w = state.widgets
    if not all(k in w for k in ("label_capa_red", "label_capa_green", "label_capa_blue")):
        return
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

def etiqueta_banda(state):
    """Return the active band label with its position, e.g. 'NIR · 5/6'."""
    idx = state.contar_imagen
    name = Settings.INDEX_LABELS[idx] if 0 <= idx < len(Settings.INDEX_LABELS) else str(idx)
    bands = getattr(state, "available_bands", None) or list(range(len(state.data_image)))
    if idx in bands:
        return f"{name} · {bands.index(idx) + 1}/{len(bands)}"
    return name


def _show_band(state):
    """Render the current band and refresh the label, dropping any index overlay."""
    w = state.widgets
    # Drop a colormapped index image left at data_image[-1] so zoom/pan operate
    # on the actual band again.
    while len(state.data_image) > 6:
        state.data_image.pop()
    state.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
    state.escala = {"x": Settings.IMG_WIDTH, "y": Settings.IMG_HEIGHT}
    mostrar_imagen(state, state.data_image[state.contar_imagen])
    w["lbl_img_selec"].configure(text=etiqueta_banda(state))


def mostrar_principal(state):
    """Reset the viewer to the principal image (band 0)."""
    if not state.data_image:
        return
    state.contar_imagen = 0
    _show_band(state)


def siguiente(state):
    """Go to the next delivered band (wraps around)."""
    if not state.data_image:
        return
    bands = state.available_bands or [0, 1, 2, 3, 4, 5]
    cur = state.contar_imagen if state.contar_imagen in bands else bands[0]
    state.contar_imagen = bands[(bands.index(cur) + 1) % len(bands)]
    _show_band(state)


def anterior(state):
    """Go to the previous delivered band (wraps around)."""
    if not state.data_image:
        return
    bands = state.available_bands or [0, 1, 2, 3, 4, 5]
    cur = state.contar_imagen if state.contar_imagen in bands else bands[0]
    state.contar_imagen = bands[(bands.index(cur) - 1) % len(bands)]
    _show_band(state)


def mostrar_mascara_sam(state):
    """Render the currently selected SAM mask in the MAIN viewer (to choose)."""
    if not state.save_mask:
        return
    w = state.widgets
    m = state.save_mask[state.contar_mask]
    mostrar_imagen(state, m["mask"])
    txt = f"Máscara SAM {state.contar_mask + 1}/{len(state.save_mask)} · score {m['score']:.3f}"
    w["lbl_img_selec"].configure(text=txt)
    lbl_seg = w.get("lbl_img_selec_seg")
    if lbl_seg is not None:
        lbl_seg.configure(text=f"Seleccionada: máscara {state.contar_mask + 1}/{len(state.save_mask)}")


def siguiente_seg(state):
    """Show the next SAM mask in the viewer."""
    if not state.save_mask:
        return
    state.contar_mask = (state.contar_mask + 1) % len(state.save_mask)
    mostrar_mascara_sam(state)


def anterior_seg(state):
    """Show the previous SAM mask in the viewer."""
    if not state.save_mask:
        return
    state.contar_mask = (state.contar_mask - 1) % len(state.save_mask)
    mostrar_mascara_sam(state)


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

    # No prompt: the user adds as many points as desired and then presses
    # "Segmentar" from the tool panel.
    img_cir_sam = state.data_image[state.contar_imagen].copy()
    for punto in state.puntos_sam:
        img_cir_sam = cv2.circle(img_cir_sam, (int(punto[0]), int(punto[1])), 10, (0, 255, 0), -1)
    img_pto_sam = img_cir_sam[lim["y_1"]: a - lim["y_2"], lim["x_1"]: b - lim["x_2"]]

    mostrar_imagen(state, img_pto_sam)
    state.input_point = np.array(state.puntos_sam)
