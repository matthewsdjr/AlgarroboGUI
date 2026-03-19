"""Load and open multispectral drone images (JPG + TIF bands)."""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import exifread
import numpy as np
from PIL import Image, ImageSequence, ImageTk

from config.settings import Settings


def abrir_imagen_visible(state, ventana_principal):
    """Open a file dialog, show a loading GIF and load all bands."""

    if not ventana_principal:
        messagebox.showerror("Error", "La ventana principal no está abierta.")
        return

    file_path = filedialog.askopenfilename(
        filetypes=[("Image file", "*.jpg *.jpeg *.png *.bmp *.gif *.tif")]
    )
    if not file_path:
        return

    state.file_path = file_path

    imagen_gif = tk.Toplevel(ventana_principal)
    imagen_gif.overrideredirect(True)
    imagen_gif.attributes("-toolwindow", 1)

    label_gif = tk.Label(imagen_gif, bg="white")
    label_gif.pack()

    gif_ancho, gif_alto = 250, 250
    x = ventana_principal.winfo_x() + (ventana_principal.winfo_width() - gif_ancho) // 2
    y = ventana_principal.winfo_y() + (ventana_principal.winfo_height() - gif_alto) // 2
    imagen_gif.geometry(f"{gif_ancho}x{gif_alto}+{x}+{y}")

    gif = Image.open(str(Settings.LOADING_GIF))
    gif_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    _show_gif(imagen_gif, label_gif, gif_frames)
    thread = threading.Thread(
        target=_abrir_mostrar, args=(state, file_path, imagen_gif)
    )
    thread.start()


def _show_gif(window, label, frames, index=0):
    frame = frames[index]
    photo = ImageTk.PhotoImage(frame)
    label.config(image=photo)
    label.image = photo
    window.after(100, lambda: _show_gif(window, label, frames, (index + 1) % len(frames)))


def extraer_coord(file_path):
    """Extract GPS latitude and longitude from EXIF data of an image file."""

    def _convertir(item):
        if "/" in item:
            num, den = map(int, item.split("/"))
            return float(num / den)
        return int(item)

    with open(file_path, "rb") as f:
        tags = exifread.process_file(f)

    lat_raw = tags["GPS GPSLatitude"].printable.strip("[]").replace(" ", "")
    lon_raw = tags["GPS GPSLongitude"].printable.strip("[]").replace(" ", "")

    lat_parts = [_convertir(x) for x in lat_raw.split(",")]
    lon_parts = [_convertir(x) for x in lon_raw.split(",")]

    latitude = lat_parts[0] + lat_parts[1] / 60 + lat_parts[2] / 3600
    longitude = lon_parts[0] + lon_parts[1] / 60 + lon_parts[2] / 3600

    return latitude, longitude


def _abrir_mostrar(state, file_path, popup):
    """Background thread: load all 6 bands and prepare the display."""
    from src.image_processing.spectral_indices import calcIndices
    from src.image_processing.image_viewer import mostrar_imagen, bandas_independientes

    w = state.widgets
    base = file_path[: -5]

    try:
        state.latitude, state.longitude = extraer_coord(file_path)
    except Exception:
        state.latitude, state.longitude = None, None

    img_principal = Image.open(base + "0.JPG").resize((Settings.IMG_WIDTH, Settings.IMG_HEIGHT))
    state.data_image = [np.array(img_principal)]

    for i in range(1, 6):
        img = Image.open(base + str(i) + ".TIF").resize(
            (Settings.IMG_WIDTH, Settings.IMG_HEIGHT)
        )
        state.data_image.append(np.array(img))

    state.a, state.b, _ = np.shape(state.data_image[0])
    state.puntos_visibles = state.data_image[0].copy()

    mostrar_imagen(state, state.data_image[0])
    w["lbl_img_selec"].configure(text=Settings.INDEX_LABELS[0])
    bandas_independientes(state, state.data_image)

    state.contorno = []
    w["cmb_1"].set("RED")
    w["cmb_2"].set("GREEN")
    w["cmb_3"].set("BLUE")
    w["cmb_1"].config(state="normal")
    w["cmb_2"].config(state="normal")
    w["cmb_3"].config(state="normal")

    state.escala = {"x": state.b, "y": state.a}
    state.limites = {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0}
    state.puntos_imagen = np.zeros((6, 2))

    popup.destroy()

    filtro_ = messagebox.askquestion("UAV", "¿Desea realizar la corrección geométrica?")
    if filtro_ == "yes":
        state.flag_correccion = 1
        w["ttk_label_imagen"].configure(cursor="plus")
    else:
        state.flag_correccion = 0
        w["ttk_label_imagen"].configure(cursor="arrow")
