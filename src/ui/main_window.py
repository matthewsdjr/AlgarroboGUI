"""Main application window: image viewer, map, segmentation panels and menus."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import customtkinter as ctk
import numpy as np
import requests
from PIL import Image, ImageTk
from tkintermapview import TkinterMapView
from scipy.io import savemat
import cv2

from config.settings import Settings

from src.image_processing.image_loader import abrir_imagen_visible
from src.image_processing.image_viewer import (
    mostrar_imagen, siguiente, anterior, siguiente_seg, anterior_seg,
    zoom_in, zoom_out, zoom_scroll, mover_mouse, mascara_presionado,
)
from src.image_processing.spectral_indices import boton_indices_event
from src.image_processing.geometric_correction import punto, correccion_imagen
from src.segmentation.manual_segmentation import mascara, roi
from src.segmentation.sam_segmentation import sam_segm_flag, sam_segm
from src.classification.alexnet_classifier import clasificar_alexnet_actualizado
from src.classification.algarrobonet_classifier import clasificar_algarrobonet_actualizado
from src.training.params_manager import ventana_configurar_parametros
from src.training.trainer_alexnet import entrenar_modelo_alexnet
from src.training.trainer_algarrobonet import entrenar_algarrobonet


def build_main_window(root, state):
    """Construct the main application window and register all widgets in state."""

    root.withdraw()

    ventana = tk.Toplevel()
    ventana.title("Software de Algarrobo")
    ventana.state("zoomed")
    try:
        ventana.iconbitmap(str(Settings.APP_ICON))
    except Exception:
        pass
    ventana.resizable(False, False)

    # Compute scale factors from the actual maximised window size.
    # The original layout was designed for 1920 × 1000 px; all hardcoded
    # dimensions below are derived from these base values so the layout
    # adapts to any screen resolution.
    ventana.update()
    ww = ventana.winfo_width()
    wh = ventana.winfo_height()
    if ww <= 1:  # safety fallback if update() hasn't resolved layout yet
        ww = ventana.winfo_screenwidth()
        wh = ventana.winfo_screenheight() - 80
    sx = ww / 1920
    sy = wh / 1000
    half = ww // 2
    img_w = int(800 * sx)
    img_h = int(650 * sy)
    bg_w = int(img_w * 1.1)
    bg_h = int(img_h * 1.1)
    btn_y = int(910 * sy)
    # Max logo height = 80 % of the white space above the centred container
    logo_max_h = max(40, int((wh - bg_h) / 2 * 0.80))
    ventana.protocol("WM_DELETE_WINDOW", root.destroy)

    w = state.widgets

    # ── Load icons ───────────────────────────────────────────────
    try:
        i_abrir = Image.open(str(Settings.ICON_IMAGEN)).resize((10, 10))
        icono_Abrir = ImageTk.PhotoImage(i_abrir)
        i_guardar = Image.open(str(Settings.ICON_GUARDAR)).resize((10, 10))
        icono_guardar = ImageTk.PhotoImage(i_guardar)
        i_pregunta = Image.open(str(Settings.ICON_PREGUNTA)).resize((10, 10))
        icono_pregunta = ImageTk.PhotoImage(i_pregunta)
        i_mapa = Image.open(str(Settings.ICON_MAP)).resize((10, 10))
        icono_mapa = ImageTk.PhotoImage(i_mapa)
        i_segm = Image.open(str(Settings.ICON_COPA)).resize((10, 10))
        icono_segm = ImageTk.PhotoImage(i_segm)
        i_correccion = Image.open(str(Settings.ICON_VALIDACION)).resize((10, 10))
        icono_correccion = ImageTk.PhotoImage(i_correccion)
        i_clasif = Image.open(str(Settings.ICON_ML)).resize((15, 15))
        icono_clasificacion = ImageTk.PhotoImage(i_clasif)

        icono_lapiz = ctk.CTkImage(Image.open(str(Settings.ICON_LAPIZ)))
        icono_aumentar = ctk.CTkImage(Image.open(str(Settings.ICON_AUMENTAR)))
        icono_disminuir = ctk.CTkImage(Image.open(str(Settings.ICON_DISMINUIR)))
        icono_siguiente = ctk.CTkImage(Image.open(str(Settings.ICON_SIGUIENTE)))
        icono_atras = ctk.CTkImage(Image.open(str(Settings.ICON_ATRAS)))
        logo_prociencia_img = Image.open(str(Settings.LOGO_PROCIENCIA_JPG))
    except Exception as e:
        messagebox.showerror("Error de carga de imágenes", f"No se pudo cargar una o más imágenes: {e}")
        ventana.destroy()
        root.quit()
        return

    # ── Menu bar ─────────────────────────────────────────────────
    barra_menu = tk.Menu(ventana)
    ventana.config(menu=barra_menu)

    # File
    archivo = tk.Menu(barra_menu, tearoff=0)
    barra_menu.add_cascade(label="Archivo", menu=archivo)
    archivo.add_command(label="Abrir Imagen", image=icono_Abrir, compound="left",
                        command=lambda: abrir_imagen_visible(state, ventana))
    archivo.add_separator()
    archivo.add_command(label="Guardar Data", image=icono_guardar, compound="left",
                        command=lambda: _guardar(state))
    archivo.add_separator()
    archivo.add_command(label="Ver Mapa", image=icono_mapa, compound="left",
                        command=lambda: _mapa(state))
    archivo.add_separator()

    # Pre-processing
    pre_proc = tk.Menu(barra_menu, tearoff=0)
    barra_menu.add_cascade(label="Pre-Procesamiento", menu=pre_proc)

    menu_seg = tk.Menu(pre_proc, tearoff=0)
    menu_seg.add_command(label="Manual Segmentation", command=lambda: mascara(state))
    menu_sam = tk.Menu(menu_seg, tearoff=0)
    menu_sam.add_command(label="Toma de Puntos", command=lambda: sam_segm_flag(state))
    menu_sam.add_separator()
    menu_sam.add_command(label="Segmentar SAM", command=lambda: sam_segm(state))
    menu_seg.add_cascade(label="SAM-Segmentation", menu=menu_sam)

    menu_corr = tk.Menu(pre_proc, tearoff=0)
    menu_corr.add_command(label="Corregir", command=lambda: correccion_imagen(state))

    pre_proc.add_cascade(label="Segmentación", menu=menu_seg, image=icono_segm, compound="left")
    pre_proc.add_cascade(label="Corrección", menu=menu_corr, image=icono_correccion, compound="left")

    # Classification
    clasif = tk.Menu(barra_menu, tearoff=0)
    barra_menu.add_cascade(label="Clasificación", menu=clasif)
    clasif.add_command(label="AlexNet", image=icono_clasificacion, compound="left",
                       command=lambda: clasificar_alexnet_actualizado(state))
    clasif.add_command(label="AlgarroboNet", image=icono_clasificacion, compound="left",
                       command=lambda: clasificar_algarrobonet_actualizado(state))

    # Training
    entren = tk.Menu(barra_menu, tearoff=0)
    barra_menu.add_cascade(label="Entrenamiento", menu=entren)
    entren.add_command(label="Parámetros de red", command=ventana_configurar_parametros)
    entren.add_separator()
    entren.add_command(label="Actualizar Database", command=lambda: _guardar_img(state))
    entren.add_separator()
    menu_red = tk.Menu(entren, tearoff=0)
    menu_red.add_command(label="AlexNet", command=entrenar_modelo_alexnet)
    menu_red.add_command(label="AlgarroboNet", command=entrenar_algarrobonet)
    entren.add_cascade(label="Entrenar", menu=menu_red)

    # Help
    ayuda = tk.Menu(barra_menu, tearoff=0)
    barra_menu.add_cascade(label="Ayuda", menu=ayuda)
    ayuda.add_command(label="Ayuda", image=icono_pregunta, compound="left")

    # ── Right panel: main image viewer ───────────────────────────
    frame_imagen = tk.Frame(ventana, width=half, height=ventana.winfo_height(), bg="white")
    frame_imagen.place(x=half, y=0)

    label_imagen_bg = ctk.CTkLabel(frame_imagen, width=bg_w, height=bg_h,
                                    corner_radius=50, bg_color="#ededed", text="")
    label_imagen_bg.place(relx=0.5, rely=0.5, anchor="center")

    label_imagen = tk.Frame(frame_imagen, width=img_w, height=img_h, bg="#ededed")
    label_imagen.place(relx=0.5, rely=0.5, anchor="center")

    lbl_img_selec = ctk.CTkLabel(label_imagen_bg, width=20, text="",
                                  font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                                  fg_color="#ededed", text_color="black")
    lbl_img_selec.place(relx=0.5, rely=0.975, anchor="center")
    w["lbl_img_selec"] = lbl_img_selec

    sty = ttk.Style()
    sty.configure("TNotebook", background="#ededed", foreground="#ededed", padding=0, borderwidth=0)
    sty.configure("TNotebook.Tab", background="#ededed", borderwidth=0)

    notebook = ttk.Notebook(label_imagen, style="TNotebook")
    ttk_label_imagen = ttk.Label(notebook, text="")
    ttk_label_imagen.pack(fill="both", expand=True)
    ttk_label_mapa = ttk.Label(notebook, text="")
    ttk_label_mapa.pack(fill="both", expand=True)
    notebook.add(ttk_label_imagen, text="Vista imagen")
    notebook.add(ttk_label_mapa, text="Mapa")

    w["ttk_label_imagen"] = ttk_label_imagen

    # Map widget
    nombre_UNF = "Universidad Nacional de Frontera Sullana"
    try:
        url = "https://nominatim.openstreetmap.org/search.php"
        params = {"q": nombre_UNF, "format": "jsonv2", "limit": 1}
        headers = {"User-Agent": "AlgarroboApp"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        data_mapa = resp.json()
        lat = float(data_mapa[0]["lat"])
        lon = float(data_mapa[0]["lon"])
    except Exception:
        lat, lon = -4.9032, -80.6827

    mapa_widget = TkinterMapView(ttk_label_mapa, width=img_w, height=img_h)
    mapa_widget.pack(fill="both", expand=True)
    mapa_widget.set_position(lat, lon, marker=True)
    mapa_widget.set_tile_server(
        "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22,
    )
    w["mapa_widget"] = mapa_widget

    notebook.place(x=0, y=0, width=img_w, height=img_h)

    # Zoom / tool buttons
    boton_disminuir = ctk.CTkButton(frame_imagen, text="", width=10, image=icono_disminuir,
                                     fg_color="Teal", command=lambda: zoom_out(state))
    boton_disminuir.image = icono_disminuir
    boton_disminuir.place(x=int(800 * sx), y=btn_y)

    boton_aumentar = ctk.CTkButton(frame_imagen, text="", width=10, image=icono_aumentar,
                                    fg_color="Teal", command=lambda: zoom_in(state))
    boton_aumentar.image = icono_aumentar
    boton_aumentar.place(x=int(850 * sx), y=btn_y)

    boton_lapiz = ctk.CTkButton(frame_imagen, text="", width=10, image=icono_lapiz,
                                 fg_color="Teal", command=lambda: roi(state))
    boton_lapiz.image = icono_lapiz
    boton_lapiz.place(x=int(900 * sx), y=btn_y)
    w["boton_lapiz"] = boton_lapiz

    boton_siguiente = ctk.CTkButton(label_imagen_bg, text="", width=7, height=5,
                                     image=icono_siguiente, fg_color="#ededed",
                                     command=lambda: siguiente(state))
    boton_siguiente.image = icono_siguiente
    boton_siguiente.place(relx=0.95, rely=0.995, anchor="se")

    boton_atras = ctk.CTkButton(label_imagen_bg, text="", width=7, height=5,
                                 image=icono_atras, fg_color="#ededed",
                                 command=lambda: anterior(state))
    boton_atras.image = icono_atras
    boton_atras.place(relx=0.05, rely=0.995, anchor="sw")

    # Logos
    pro_w = int(logo_max_h * 210 / 140)
    logo_prociencia_resized = logo_prociencia_img.resize((pro_w, logo_max_h), Image.LANCZOS)
    foto_prociencia = ImageTk.PhotoImage(logo_prociencia_resized)
    prociencia = tk.Label(frame_imagen, image=foto_prociencia, bg="white")
    prociencia.image = foto_prociencia
    prociencia.place(x=half - pro_w - int(10 * sx), y=0)

    unf_w = int(logo_max_h * 103 / 131)
    logo_unf_resized = Image.open(str(Settings.LOGO_UNF)).resize((unf_w, logo_max_h), Image.LANCZOS)
    foto_unf = ImageTk.PhotoImage(logo_unf_resized)
    logo_principal = tk.Label(frame_imagen, image=foto_unf, bg="white")
    logo_principal.image = foto_unf
    logo_principal.place(x=int(10 * sx), y=0)

    # Mouse bindings
    ttk_label_imagen.bind("<ButtonPress-3>", lambda e: punto(state, e))
    ttk_label_imagen.bind("<MouseWheel>", lambda e: zoom_scroll(state, e))
    ttk_label_imagen.bind("<B1-Motion>", lambda e: mover_mouse(state, e))
    ttk_label_imagen.bind("<ButtonPress-1>", lambda e: mascara_presionado(state, e))

    # ── Left panel: band thumbnails + segmentation ───────────────
    frame_indp = tk.Frame(ventana, width=half, height=ventana.winfo_height(), bg="white")
    frame_indp.place(x=0, y=0)

    label_bg2 = ctk.CTkLabel(frame_indp, width=bg_w, height=bg_h,
                              corner_radius=50, bg_color="#ededed", text="")
    label_bg2.place(relx=0.5, rely=0.5, anchor="center")

    label_indp2 = tk.Frame(frame_indp, width=img_w, height=img_h, bg="#ededed")
    label_indp2.place(relx=0.5, rely=0.5, anchor="center")

    # Title labels created AFTER the containers so they sit on top (higher z-order).
    texto_proj = '"Redes Convolucionales y Combinacionales de Bandas Espectrales'
    texto_2 = 'e Índices Vegetativos para Identificación de Árboles Plus de Algarrobo (Prosopis Pallida): '
    texto_3 = 'Aplicación de Técnicas Deep Learning e Imágenes Multiespectrales"'

    title_font_size = max(10, int(13 * sx))
    for i, txt in enumerate([texto_proj, texto_2, texto_3]):
        ctk.CTkLabel(
            frame_indp, width=20, text=txt,
            font=ctk.CTkFont(family="Arial", size=title_font_size, weight="bold"),
            fg_color="white", text_color="black",
            wraplength=int(half * 0.92),
        ).place(relx=0.5, rely=0.05 + i * 0.03, anchor="center")

    notebook_2 = ttk.Notebook(label_indp2, style="TNotebook")
    ttk_label_indep = ttk.Label(notebook_2, text="")
    ttk_label_indep.pack(fill="both", expand=True)
    ttk_label_segm = ttk.Label(notebook_2, text="")
    ttk_label_segm.pack(fill="both", expand=True)
    notebook_2.add(ttk_label_indep, text="Independientes")
    notebook_2.add(ttk_label_segm, text="Segmentación")
    notebook_2.place(x=0, y=0, width=img_w, height=img_h)

    # Segmentation panel
    bg_segm = 235 * np.ones((img_h, img_w), np.uint8)
    bg_f_segm = ImageTk.PhotoImage(image=Image.fromarray(bg_segm))
    ttk_label_segm.configure(image=bg_f_segm)
    ttk_label_segm.image = bg_f_segm

    label_img_segm = tk.Label(ttk_label_segm, bg="#ededed")
    label_img_segm.place(relx=0.5, rely=0.55, anchor="center")
    w["label_img_segm"] = label_img_segm

    arr_segm = 225 * np.ones((int(450 * sy), int(550 * sx)), np.uint8)
    foto_segm = ImageTk.PhotoImage(image=Image.fromarray(arr_segm))
    label_img_segm.configure(image=foto_segm)
    label_img_segm.image = foto_segm

    boton_sig_seg = ctk.CTkButton(ttk_label_segm, text="", width=7, height=5,
                                   image=icono_siguiente, fg_color="#ededed",
                                   command=lambda: siguiente_seg(state))
    boton_sig_seg.image = icono_siguiente
    boton_sig_seg.place(x=int(720 * sx), y=int(300 * sy), anchor="se")

    boton_atr_seg = ctk.CTkButton(ttk_label_segm, text="", width=7, height=5,
                                   image=icono_atras, fg_color="#ededed",
                                   command=lambda: anterior_seg(state))
    boton_atr_seg.image = icono_atras
    boton_atr_seg.place(x=int(80 * sx), y=int(300 * sy), anchor="sw")

    lbl_img_selec_seg = ctk.CTkLabel(ttk_label_segm, width=20, text="",
                                      font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                                      fg_color="#ededed", text_color="black")
    lbl_img_selec_seg.place(x=int(400 * sx), y=int(40 * sy), anchor="center")
    w["lbl_img_selec_seg"] = lbl_img_selec_seg

    # Independent band panel
    bg_indep = 230 * np.ones((img_h, img_w), np.uint8)
    bg_f = ImageTk.PhotoImage(image=Image.fromarray(bg_indep))
    ttk_label_indep.configure(image=bg_f)
    ttk_label_indep.image = bg_f

    label_capa_red = tk.Label(ttk_label_indep, bg="red")
    label_capa_red.place(relx=0.5, rely=0.2, anchor="center")
    w["label_capa_red"] = label_capa_red

    im_placeholder = 230 * np.ones((int(215 * sy), int(250 * sx)), np.uint8)
    _set_placeholder(label_capa_red, im_placeholder)

    label_capa_green = tk.Label(ttk_label_indep, bg="green")
    label_capa_green.place(relx=0.2, rely=0.6, anchor="center")
    w["label_capa_green"] = label_capa_green
    _set_placeholder(label_capa_green, im_placeholder)

    label_capa_blue = tk.Label(ttk_label_indep, bg="blue")
    label_capa_blue.place(relx=0.8, rely=0.6, anchor="center")
    w["label_capa_blue"] = label_capa_blue
    _set_placeholder(label_capa_blue, im_placeholder)

    # Combo boxes for index selection
    lista = Settings.INDEX_LABELS
    cmb_1 = ttk.Combobox(ttk_label_indep, values=lista, state="readonly")
    cmb_1.place(relx=0.5, rely=0.35, anchor="center")
    cmb_1.bind("<<ComboboxSelected>>", lambda e: boton_indices_event(state, e))
    cmb_1.config(state="disabled", justify="center")
    w["cmb_1"] = cmb_1

    cmb_2 = ttk.Combobox(ttk_label_indep, values=lista, state="readonly")
    cmb_2.place(relx=0.2, rely=0.75, anchor="center")
    cmb_2.bind("<<ComboboxSelected>>", lambda e: boton_indices_event(state, e))
    cmb_2.config(state="disabled", justify="center")
    w["cmb_2"] = cmb_2

    cmb_3 = ttk.Combobox(ttk_label_indep, values=lista, state="readonly")
    cmb_3.place(relx=0.8, rely=0.75, anchor="center")
    cmb_3.bind("<<ComboboxSelected>>", lambda e: boton_indices_event(state, e))
    cmb_3.config(state="disabled", justify="center")
    w["cmb_3"] = cmb_3


def _set_placeholder(label, array):
    img = ImageTk.PhotoImage(image=Image.fromarray(array))
    label.configure(image=img)
    label.image = img


# ── Helper actions ───────────────────────────────────────────────────────────

def _guardar(state):
    """Save indices + segmentation mask to .mat file."""
    num_channels = 27
    if state.save_mask:
        num_channels += 3 * len(state.save_mask)

    save = np.zeros((1300, 1600, num_channels), np.float16)
    save[:, :, 0:24] = state.Indices
    if state.imagen_segementada is not None:
        save[:, :, 24:27] = state.imagen_segementada

    save[np.isnan(save)] = 0
    save[np.isinf(save)] = 1

    file_save = filedialog.asksaveasfilename(
        defaultextension=".mat",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
    )
    if file_save:
        savemat(file_save, {"data": save}, do_compression=True)
        messagebox.showinfo("Aviso", "Guardado con éxito")


def _guardar_img(state):
    """Save current segmented mask as JPEG for dataset building."""
    file_save = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
    )
    if file_save and state.save_mask:
        img_ = state.save_mask[state.contar_mask]["mask"]
        img_resized = cv2.resize(img_, (227, 227), interpolation=cv2.INTER_AREA)
        Image.fromarray(img_resized).save(file_save, format="JPEG", quality=95)
        messagebox.showinfo("Aviso", "Imagen guardada con éxito")


def _mapa(state):
    """Pan the map widget to the GPS coordinates extracted from the loaded image."""
    if not state.file_path:
        messagebox.showinfo("Aviso", "Adjunte una imagen primero.")
        return
    if state.latitude is None or state.longitude is None:
        messagebox.showwarning("Aviso", "No se encontraron coordenadas GPS en la imagen.")
        return

    mapa_widget = state.widgets.get("mapa_widget")
    if mapa_widget:
        mapa_widget.set_position(-state.latitude, -state.longitude, marker=True)
        mapa_widget.set_zoom(18)
