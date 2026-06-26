"""Load and open multispectral drone images (JPG + TIF bands)."""

import os
import re
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


def _es_formato_dji(file_path):
    """Return True if the file matches the DJI multispectral naming convention (e.g. DJI_20260512095234_0001_D.JPG)."""
    filename = os.path.basename(file_path)
    return bool(re.match(r"DJI_\d+_\d+_D\.(JPG|JPEG)$", filename, re.IGNORECASE))


def _buscar_banda_dji(folder, seq, suffix):
    """Find a DJI band file by sequence number, ignoring the timestamp part.

    DJI saves the RGB and multispectral TIFs one second apart, so their
    timestamps differ while the sequence number stays the same.
    Matches files of the form: DJI_<timestamp>_<seq>_<suffix>
    """
    pattern = re.compile(
        rf"DJI_\d+_{re.escape(seq)}_{re.escape(suffix)}$", re.IGNORECASE
    )
    matches = [f for f in os.listdir(folder) if pattern.match(f)]
    return os.path.join(folder, matches[0]) if matches else None


def _cargar_bandas_dji(file_path):
    """Load all 6 bands for the DJI multispectral format and return them as a list of arrays.

    Band layout (same as LPP format):
      [0] RGB composite (JPG), [1] R, [2] G, [3] B, [4] RE, [5] NIR
    Matching is done by sequence number (e.g. "0009"), not by timestamp,
    because DJI saves the RGB and the TIF bands with slightly different timestamps.
    Blue TIF is optional: if MS_B.TIF is absent, B is extracted from the RGB image.
    """
    folder = os.path.dirname(os.path.abspath(file_path))
    filename = os.path.basename(file_path)

    # Extract sequence number: DJI_<timestamp>_<seq>_D.JPG
    m = re.match(r"DJI_\d+_(\d+)_D\.(JPG|JPEG)$", filename, re.IGNORECASE)
    seq = m.group(1)  # e.g. "0009"

    img_rgb = Image.open(file_path).resize((Settings.IMG_WIDTH, Settings.IMG_HEIGHT))
    rgb_array = np.array(img_rgb)  # uint8 (H, W, 3)

    bands = [None] * 6
    bands[0] = rgb_array

    for suffix, idx in (("MS_R.TIF", 1), ("MS_G.TIF", 2), ("MS_RE.TIF", 4), ("MS_NIR.TIF", 5)):
        band_path = _buscar_banda_dji(folder, seq, suffix)
        if band_path is None:
            raise FileNotFoundError(
                f"No se encontró la banda '{suffix}' para la secuencia '{seq}' en:\n{folder}"
            )
        img = Image.open(band_path).resize((Settings.IMG_WIDTH, Settings.IMG_HEIGHT))
        bands[idx] = np.array(img)

    # Blue band: use MS_B.TIF if present, otherwise extract from RGB for index computation
    b_path = _buscar_banda_dji(folder, seq, "MS_B.TIF")
    if b_path is not None:
        img_b = Image.open(b_path).resize((Settings.IMG_WIDTH, Settings.IMG_HEIGHT))
        bands[3] = np.array(img_b)
        tiene_azul = True
    else:
        # Scale uint8 blue channel to uint16 range so spectral indices can still use it
        bands[3] = (rgb_array[:, :, 2].astype(np.uint16) * 257)
        tiene_azul = False

    return bands, tiene_azul


def _abrir_mostrar(state, file_path, popup):
    """Background thread: load all 6 bands and prepare the display."""
    from src.image_processing.spectral_indices import calcIndices, graficar_indices
    from src.image_processing.image_viewer import mostrar_imagen, bandas_independientes

    w = state.widgets

    try:
        state.latitude, state.longitude = extraer_coord(file_path)
    except Exception:
        state.latitude, state.longitude = None, None

    if _es_formato_dji(file_path):
        state.data_image, state.tiene_banda_azul = _cargar_bandas_dji(file_path)
    else:
        # LPP legacy format: user selects e.g. LPP_100.JPG
        # base = file_path[:-5] strips "0.JPG" → loads LPP_100.JPG + LPP_101..105.TIF
        state.tiene_banda_azul = True
        base = file_path[:-5]
        img_principal = Image.open(base + "0.JPG").resize((Settings.IMG_WIDTH, Settings.IMG_HEIGHT))
        state.data_image = [np.array(img_principal)]
        for i in range(1, 6):
            img = Image.open(base + str(i) + ".TIF").resize(
                (Settings.IMG_WIDTH, Settings.IMG_HEIGHT)
            )
            state.data_image.append(np.array(img))

    state.a, state.b, _ = np.shape(state.data_image[0])
    state.puntos_visibles = state.data_image[0].copy()

    calcIndices(state, state.data_image)
    mostrar_imagen(state, state.data_image[0])
    w["lbl_img_selec"].configure(text=Settings.INDEX_LABELS[0])
    bandas_independientes(state, state.data_image)

    state.contorno = []
    cmb_3_default = "NIR" if not state.tiene_banda_azul else "BLUE"
    w["cmb_1"].set("RED")
    w["cmb_2"].set("GREEN")
    w["cmb_3"].set(cmb_3_default)
    w["cmb_1"].config(state="normal")
    w["cmb_2"].config(state="normal")
    w["cmb_3"].config(state="normal")

    p = Settings.INDEX_LABELS.index("RED") - 1
    s = Settings.INDEX_LABELS.index("GREEN") - 1
    t = Settings.INDEX_LABELS.index(cmb_3_default) - 1
    graficar_indices(state, state.Indices, p, s, t)

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
