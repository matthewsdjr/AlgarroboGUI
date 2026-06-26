"""Main application window.

Modern single-viewer layout:
  ┌──────────────────────────────────────────────────────────┐
  │  [UNF]        institutional title (centred)   [Prociencia] │  fixed header
  ├──────────────────────────────────────────────────────────┤
  │  Abrir · Mapa · Bandas · Índices · Segmentar · Clasificar  │  top toolbar
  ├──────────────────────────────────────────────────────────┤
  │   ◄   [   single 800×650 viewer   ]   ►    │  tool panel    │  content
  └──────────────────────────────────────────────────────────┘

All processing logic keeps reading/writing widgets through ``state.widgets``
by key, and the main image is still rendered at 800×650 px, so the coordinate
math in the processing modules is preserved unchanged.
"""

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
    mostrar_imagen, mostrar_principal, siguiente, anterior, siguiente_seg, anterior_seg,
    zoom_scroll, mover_mouse, mascara_presionado,
)
from src.image_processing.spectral_indices import (
    mostrar_indice, hover_indice, colorbar_image, INDEX_DEFS,
)
from src.image_processing.geometric_correction import punto, correccion_imagen
from src.segmentation.manual_segmentation import mascara
from src.segmentation.sam_segmentation import sam_segm_flag, sam_segm
from src.classification.alexnet_classifier import clasificar_alexnet_actualizado
from src.classification.algarrobonet_classifier import clasificar_algarrobonet_actualizado
from src.training.params_manager import ventana_configurar_parametros
from src.training.trainer_alexnet import entrenar_modelo_alexnet
from src.training.trainer_algarrobonet import entrenar_algarrobonet


# ── Visual identity ───────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#f4f6f8",   # app background
    "header":    "#ffffff",   # header / toolbar background
    "panel":     "#ffffff",   # contextual tool panel
    "viewer":    "#e9edf0",   # viewer canvas backdrop
    "primary":   "#0f766e",   # teal accent
    "primary_d": "#0b5a54",   # darker teal (hover / active)
    "text":      "#1f2933",
    "muted":     "#6b7280",
}

# Logical render size of the main viewer (must stay 800×650: the coordinate
# transforms in the processing modules are hardcoded to these values).
DISP_W = int(Settings.IMG_WIDTH * (1 - Settings.DISPLAY_FACTOR))   # 800
DISP_H = int(Settings.IMG_HEIGHT * (1 - Settings.DISPLAY_FACTOR))  # 650

APP_VERSION = "2.0"

# Full institutional title (kept verbatim from the project).
TITULO = (
    '"Redes Convolucionales y Combinacionales de Bandas Espectrales e Índices '
    "Vegetativos para Identificación de Árboles Plus de Algarrobo (Prosopis "
    'Pallida): Aplicación de Técnicas Deep Learning e Imágenes Multiespectrales"'
)

# User-facing help, shown in the Ayuda window. Each item: (emoji, title, [lines]).
HELP_SECTIONS = [
    ("👋", "Bienvenido", [
        "Algarrobo analiza imágenes multiespectrales de dron y clasifica árboles de "
        "algarrobo (Prosopis pallida).",
        "Esta guía resume cómo usar cada herramienta de la interfaz.",
    ]),
    ("🧭", "La interfaz", [
        "Encabezado superior con los logos institucionales y el título del proyecto.",
        "Barra de herramientas: Abrir imagen, Bandas, Índices, Mapa, Segmentación, "
        "Clasificación, Corrección, Entrenamiento y Guardar.",
        "Visor central donde se muestra la imagen, los índices, el mapa o las máscaras.",
        "Panel lateral que cambia según la herramienta activa.",
    ]),
    ("📂", "Abrir una imagen", [
        "Pulsa «Abrir imagen» y selecciona la toma del dron.",
        "Las bandas se cargan solas y se muestra la imagen principal (compuesta RGB).",
    ]),
    ("🛰", "Navegación de bandas", [
        "Solo se muestran las bandas que entregó el dron; un contador indica la "
        "posición (p. ej. RED · 2/6).",
        "Cambia de banda con las flechas en pantalla o con el teclado (← / →).",
        "Haz zoom con la rueda del mouse y desplázate arrastrando sobre la imagen.",
    ]),
    ("🌿", "Índices de vegetación", [
        "Abre la pestaña Índices y elige un índice del menú.",
        "Se dibuja con un mapa de color (rojo→verde) y una leyenda con la escala.",
        "Pasa el cursor sobre la imagen para ver el valor del índice en cada píxel.",
        "Disponibles: NDVI, GNDVI, NDRE, NGRDI, NGBDI, RVI y SCCI.",
    ]),
    ("✂", "Segmentación", [
        "Manual: pulsa «Segmentación manual», traza el contorno arrastrando el cursor "
        "y pulsa «Crear máscara».",
        "SAM: pulsa «Tomar puntos», haz clic sobre el objeto (uno o varios) y pulsa "
        "«Segmentar SAM».",
        "Con SAM se generan 3 máscaras; recórrelas con ◄ ► y deja visible la que quieras usar.",
    ]),
    ("🧠", "Clasificación", [
        "Primero segmenta un objeto (manual o SAM).",
        "Elige el modelo: AlexNet o AlgarroboNet.",
        "El resultado aparece como una tarjeta con la clase (Plus / No Plus) y la "
        "confianza por clase.",
    ]),
    ("📍", "Mapa", [
        "Muestra la ubicación de la toma sobre un mapa satelital, según las "
        "coordenadas GPS de la imagen.",
        "Usa «Volver a la imagen» para regresar a la vista de bandas.",
    ]),
    ("🎯", "Corrección de bandas", [
        "Es opcional y sirve para alinear todas las bandas.",
        "Activa la selección de puntos y marca con clic derecho el mismo punto de "
        "referencia en cada banda.",
        "Pulsa «Corregir bandas» para alinearlas.",
    ]),
    ("💾", "Guardar y entrenamiento", [
        "Guardar: exporta los índices y la máscara a un archivo .mat.",
        "Entrenamiento: configura los parámetros, actualiza la base de datos y "
        "reentrena los modelos.",
    ]),
]


def build_main_window(root, state):
    """Construct the main application window and register all widgets in state."""

    ctk.set_appearance_mode("light")
    root.withdraw()

    ventana = tk.Toplevel()
    ventana.title("Software de Algarrobo")
    ventana.configure(bg=COLORS["bg"])
    try:
        ventana.iconbitmap(str(Settings.APP_ICON))
    except Exception:
        pass

    # Open as a centred, resizable window (no longer locked full-screen). The
    # user can freely resize or maximise it; the 800×650 viewer stays centred.
    sw, sh = ventana.winfo_screenwidth(), ventana.winfo_screenheight()
    ww = min(1600, int(sw * 0.92))
    wh = min(980, int(sh * 0.92))
    x = max(0, (sw - ww) // 2)
    y = max(0, (sh - wh) // 2 - 16)
    ventana.geometry(f"{ww}x{wh}+{x}+{y}")
    ventana.minsize(min(1180, sw - 20), min(880, sh - 40))
    ventana.resizable(True, True)
    ventana.protocol("WM_DELETE_WINDOW", root.destroy)

    ventana.update()
    logo_h = min(110, max(64, int(wh * 0.10)))

    w = state.widgets

    # ── Load assets (logos + icons) ──────────────────────────────
    icons = {}
    try:
        unf_w = int(logo_h * 103 / 131)
        unf_img = Image.open(str(Settings.LOGO_UNF)).resize((unf_w, logo_h), Image.LANCZOS)
        icons["unf"] = ImageTk.PhotoImage(unf_img)

        pro_w = int(logo_h * 210 / 140)
        pro_img = Image.open(str(Settings.LOGO_PROCIENCIA_JPG)).resize((pro_w, logo_h), Image.LANCZOS)
        icons["prociencia"] = ImageTk.PhotoImage(pro_img)

        icons["siguiente"] = ctk.CTkImage(Image.open(str(Settings.ICON_SIGUIENTE)), size=(26, 26))
        icons["atras"] = ctk.CTkImage(Image.open(str(Settings.ICON_ATRAS)), size=(26, 26))
        icons["lapiz"] = ctk.CTkImage(Image.open(str(Settings.ICON_LAPIZ)), size=(18, 18))
        icons["aumentar"] = ctk.CTkImage(Image.open(str(Settings.ICON_AUMENTAR)), size=(18, 18))
        icons["disminuir"] = ctk.CTkImage(Image.open(str(Settings.ICON_DISMINUIR)), size=(18, 18))
    except Exception as e:
        messagebox.showerror("Error de carga de imágenes",
                             f"No se pudo cargar una o más imágenes: {e}")
        ventana.destroy()
        root.quit()
        return
    w["_icons"] = icons  # keep references so Tk does not garbage-collect them

    # ── Root grid: header / toolbar / content ────────────────────
    ventana.grid_rowconfigure(2, weight=1)
    ventana.grid_columnconfigure(0, weight=1)

    _build_header(ventana, icons, logo_h)
    _build_toolbar(ventana, state)
    _build_content(ventana, state, icons)
    _build_footer(ventana)

    # ── Keyboard navigation (← / →) ──────────────────────────────
    ventana.bind("<Left>", lambda e: _nav_band(state, anterior))
    ventana.bind("<Right>", lambda e: _nav_band(state, siguiente))
    ventana.focus_set()

    # Default tool view.
    _select_tool(state, "bandas")


# ── Layout sections ───────────────────────────────────────────────────────────

def _build_header(ventana, icons, logo_h):
    """Fixed institutional header: UNF left, title centred, Prociencia right."""
    header = tk.Frame(ventana, bg=COLORS["header"], height=logo_h + 18)
    header.grid(row=0, column=0, sticky="ew")
    header.grid_propagate(False)
    header.grid_columnconfigure(0, weight=0)
    header.grid_columnconfigure(1, weight=1)
    header.grid_columnconfigure(2, weight=0)

    tk.Label(header, image=icons["unf"], bg=COLORS["header"]).grid(
        row=0, column=0, padx=20, pady=8, sticky="w")

    ctk.CTkLabel(
        header, text=TITULO,
        font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
        text_color=COLORS["text"], fg_color=COLORS["header"],
        wraplength=int(ventana.winfo_width() * 0.55), justify="center",
    ).grid(row=0, column=1, sticky="nsew", padx=10)

    tk.Label(header, image=icons["prociencia"], bg=COLORS["header"]).grid(
        row=0, column=2, padx=20, pady=8, sticky="e")

    tk.Frame(ventana, bg="#dfe3e6", height=1).grid(row=1, column=0, sticky="ew")


def _build_footer(ventana):
    """Fixed footer with the interface version and institutional rights."""
    footer = tk.Frame(ventana, bg=COLORS["header"])
    footer.grid(row=3, column=0, sticky="ew")
    tk.Frame(footer, bg="#dfe3e6", height=1).pack(fill="x")
    ctk.CTkLabel(
        footer,
        text=f"Versión {APP_VERSION}  ·  © Universidad Nacional de Frontera",
        font=ctk.CTkFont(family="Segoe UI", size=10),
        text_color=COLORS["muted"], fg_color=COLORS["header"],
    ).pack(pady=3)


def _build_toolbar(ventana, state):
    """Top toolbar with all tools; each one renders inside the single viewer."""
    bar = tk.Frame(ventana, bg=COLORS["header"], height=58)
    bar.grid(row=1, column=0, sticky="ew")
    bar.grid_propagate(False)

    inner = tk.Frame(bar, bg=COLORS["header"])
    inner.pack(side="left", padx=16, pady=10)

    buttons = {}

    def add(key, text, command, accent=False):
        b = ctk.CTkButton(
            inner, text=text, command=command, width=10, height=36,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=COLORS["primary"] if accent else "#e7ebee",
            hover_color=COLORS["primary_d"] if accent else "#d6dbdf",
            text_color="white" if accent else COLORS["text"],
        )
        b.pack(side="left", padx=4)
        buttons[key] = b
        return b

    add("abrir", "📂  Abrir imagen",
        lambda: abrir_imagen_visible(state, state.widgets["_ventana"]), accent=True)

    _sep(inner)
    add("bandas", "🛰  Bandas", lambda: _select_tool(state, "bandas"))
    add("indices", "🌿  Índices", lambda: _select_tool(state, "indices"))
    add("mapa", "📍  Mapa", lambda: _select_tool(state, "mapa"))
    _sep(inner)
    add("segmentacion", "✂  Segmentación", lambda: _select_tool(state, "segmentacion"))
    add("clasificacion", "🧠  Clasificación", lambda: _select_tool(state, "clasificacion"))
    add("correccion", "🎯  Corrección", lambda: _select_tool(state, "correccion"))
    add("entrenamiento", "⚙  Entrenamiento", lambda: _select_tool(state, "entrenamiento"))
    _sep(inner)
    add("guardar", "💾  Guardar", lambda: _guardar(state))
    add("ayuda", "❓  Ayuda", lambda: _mostrar_ayuda(state))

    state.widgets["_toolbar_buttons"] = buttons


def _sep(parent):
    tk.Frame(parent, bg="#d6dbdf", width=1, height=30).pack(side="left", padx=8, fill="y")


def _build_content(ventana, state, icons):
    """Central area: single viewer (image / map router) + contextual tool panel."""
    w = state.widgets
    w["_ventana"] = ventana

    content = tk.Frame(ventana, bg=COLORS["bg"])
    content.grid(row=2, column=0, sticky="nsew")
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)   # viewer
    content.grid_columnconfigure(1, weight=0)   # tool panel

    # ── Viewer container (stacked views: image / map) ────────────
    viewer = tk.Frame(content, bg=COLORS["viewer"])
    viewer.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

    view_image = tk.Frame(viewer, bg=COLORS["viewer"])
    view_map = tk.Frame(viewer, bg=COLORS["viewer"])
    for v in (view_image, view_map):
        v.place(relx=0, rely=0, relwidth=1, relheight=1)
    w["_view_image"] = view_image
    w["_view_map"] = view_map

    _build_image_view(view_image, state, icons)
    _build_map_view(view_map, state)

    view_image.tkraise()

    # ── Contextual tool panel ────────────────────────────────────
    panel = ctk.CTkFrame(content, width=320, corner_radius=14, fg_color=COLORS["panel"])
    panel.grid(row=0, column=1, sticky="ns", padx=(8, 16), pady=16)
    panel.grid_propagate(False)
    w["_tool_panel"] = panel

    # Hidden compatibility labels. The geometric-correction routine still calls
    # bandas_independientes() to refresh per-band thumbnails; the single-viewer
    # layout has no left thumbnail panel, so these targets are kept off-screen.
    compat = tk.Frame(viewer)  # never placed → invisible
    for key in ("label_capa_red", "label_capa_green", "label_capa_blue"):
        w[key] = tk.Label(compat)


def _build_image_view(parent, state, icons):
    """The 800×650 image viewer with on-screen navigation arrows + band label."""
    w = state.widgets

    holder = tk.Frame(parent, bg=COLORS["viewer"])
    holder.place(relx=0.5, rely=0.5, anchor="center")
    # Reserve the side columns so the image stays centred even when the arrows
    # are hidden (grid_remove keeps the column width via minsize).
    holder.grid_columnconfigure(0, minsize=54)
    holder.grid_columnconfigure(2, minsize=54)

    # Row layout: [◄] [ image card ] [►] — arrows sit right next to the image.
    # Their visibility is toggled per tool by _set_viewer_controls (only Bandas /
    # Corrección allow changing the band).
    btn_prev = ctk.CTkButton(holder, text="", image=icons["atras"], width=44, height=44,
                             corner_radius=22, fg_color=COLORS["primary"],
                             hover_color=COLORS["primary_d"],
                             command=lambda: _nav_band(state, anterior))
    btn_prev.grid(row=0, column=0, padx=(0, 10))
    w["_btn_prev"] = btn_prev

    # Rounded backdrop behind the image.
    card = ctk.CTkFrame(holder, corner_radius=18, fg_color="#ffffff",
                        width=DISP_W + 28, height=DISP_H + 28)
    card.grid(row=0, column=1)
    card.grid_propagate(False)

    btn_next = ctk.CTkButton(holder, text="", image=icons["siguiente"], width=44, height=44,
                             corner_radius=22, fg_color=COLORS["primary"],
                             hover_color=COLORS["primary_d"],
                             command=lambda: _nav_band(state, siguiente))
    btn_next.grid(row=0, column=2, padx=(10, 0))
    w["_btn_next"] = btn_next

    ttk_label_imagen = tk.Label(card, bg="#ffffff", bd=0)
    ttk_label_imagen.place(relx=0.5, rely=0.5, anchor="center", width=DISP_W, height=DISP_H)
    w["ttk_label_imagen"] = ttk_label_imagen

    _set_placeholder(ttk_label_imagen, 235 * np.ones((DISP_H, DISP_W), np.uint8))

    # Band / view label (under the image card).
    lbl_img_selec = ctk.CTkLabel(
        holder, text="Abra una imagen para comenzar",
        font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
        text_color=COLORS["text"], fg_color=COLORS["viewer"],
    )
    lbl_img_selec.grid(row=1, column=0, columnspan=3, pady=(8, 0))
    w["lbl_img_selec"] = lbl_img_selec

    # Mouse bindings (same as the original viewer).
    ttk_label_imagen.bind("<ButtonPress-3>", lambda e: punto(state, e))
    ttk_label_imagen.bind("<MouseWheel>", lambda e: zoom_scroll(state, e))
    ttk_label_imagen.bind("<B1-Motion>", lambda e: mover_mouse(state, e))
    ttk_label_imagen.bind("<ButtonPress-1>", lambda e: mascara_presionado(state, e))
    # Live per-pixel index readout (only active while the Índices tool is open).
    ttk_label_imagen.bind("<Motion>", lambda e: hover_indice(state, e))


def _build_map_view(parent, state):
    """Embedded satellite map (shown inside the viewer, not a new window)."""
    w = state.widgets

    bar = tk.Frame(parent, bg=COLORS["viewer"])
    bar.pack(fill="x", padx=10, pady=(10, 6))
    ctk.CTkButton(bar, text="←  Volver a la imagen", width=10, height=34,
                  fg_color=COLORS["primary"], hover_color=COLORS["primary_d"],
                  command=lambda: _select_tool(state, "bandas")).pack(side="left")
    ctk.CTkLabel(bar, text="Ubicación de la toma",
                 font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                 text_color=COLORS["text"], fg_color=COLORS["viewer"]).pack(side="left", padx=14)

    map_holder = tk.Frame(parent, bg=COLORS["viewer"])
    map_holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # Initial centre: Universidad Nacional de Frontera (Sullana).
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search.php",
            params={"q": "Universidad Nacional de Frontera Sullana",
                    "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "AlgarroboApp"}, timeout=5,
        )
        data_mapa = resp.json()
        lat, lon = float(data_mapa[0]["lat"]), float(data_mapa[0]["lon"])
    except Exception:
        lat, lon = -4.9032, -80.6827

    mapa_widget = TkinterMapView(map_holder, corner_radius=12)
    mapa_widget.pack(fill="both", expand=True)
    mapa_widget.set_position(lat, lon, marker=True)
    mapa_widget.set_tile_server(
        "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22,
    )
    w["mapa_widget"] = mapa_widget


# ── Tool router ───────────────────────────────────────────────────────────────

def _select_tool(state, tool):
    """Activate a tool: switch the viewer view and rebuild the contextual panel."""
    w = state.widgets
    w["_active_tool"] = tool

    # View switching (map vs image).
    if tool == "mapa":
        _abrir_mapa(state)
        w["_view_map"].tkraise()
    else:
        w["_view_image"].tkraise()
        # Bandas, Segmentación and Corrección always open on the principal image.
        if tool in ("bandas", "segmentacion", "correccion"):
            mostrar_principal(state)

    # Arrows only where changing band makes sense (Bandas / Corrección).
    _set_viewer_controls(state, arrows=tool in ("bandas", "correccion"))

    # Highlight the active toolbar button.
    for key, btn in w.get("_toolbar_buttons", {}).items():
        if key in ("abrir", "guardar", "ayuda"):
            continue
        active = (key == tool)
        btn.configure(
            fg_color=COLORS["primary"] if active else "#e7ebee",
            text_color="white" if active else COLORS["text"],
        )

    _build_tool_panel(state, tool)


def _set_viewer_controls(state, arrows=False):
    """Show/hide the on-screen band navigation arrows per tool."""
    w = state.widgets
    if arrows:
        w["_btn_prev"].grid()
        w["_btn_next"].grid()
    else:
        w["_btn_prev"].grid_remove()
        w["_btn_next"].grid_remove()


def _clear_panel(panel):
    for child in panel.winfo_children():
        child.destroy()


def _panel_title(panel, text):
    ctk.CTkLabel(panel, text=text,
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                 text_color=COLORS["text"], anchor="w").pack(fill="x", padx=18, pady=(18, 6))
    tk.Frame(panel, bg="#e3e7ea", height=1).pack(fill="x", padx=18, pady=(0, 8))


def _panel_hint(panel, text):
    ctk.CTkLabel(panel, text=text, justify="left", anchor="w", wraplength=270,
                 font=ctk.CTkFont(family="Segoe UI", size=12),
                 text_color=COLORS["muted"]).pack(fill="x", padx=18, pady=(0, 10))


def _panel_button(panel, text, command):
    ctk.CTkButton(panel, text=text, command=command, height=38, corner_radius=8,
                  font=ctk.CTkFont(family="Segoe UI", size=13),
                  fg_color=COLORS["primary"], hover_color=COLORS["primary_d"]
                  ).pack(fill="x", padx=18, pady=5)


def _build_tool_panel(state, tool):
    panel = state.widgets["_tool_panel"]
    _clear_panel(panel)

    if tool == "bandas":
        _panel_title(panel, "🛰  Bandas")
        _panel_hint(panel, "Navega entre las bandas entregadas por el dron con las "
                           "flechas en pantalla o con ← / → del teclado. Usa la rueda "
                           "del mouse para zoom y arrastra para desplazar.")

    elif tool == "indices":
        _build_panel_indices(state, panel)

    elif tool == "mapa":
        _panel_title(panel, "📍  Mapa")
        _panel_hint(panel, "El mapa muestra la ubicación obtenida de las coordenadas "
                           "GPS de la imagen. Pulsa «Volver a la imagen» para regresar "
                           "a la vista de bandas.")

    elif tool == "segmentacion":
        _build_panel_segmentacion(state, panel)

    elif tool == "clasificacion":
        _build_panel_clasificacion(state, panel)

    elif tool == "correccion":
        _build_panel_correccion(state, panel)

    elif tool == "entrenamiento":
        _build_panel_entrenamiento(state, panel)


# ── Tool panels (interim wiring to existing logic) ────────────────────────────

def _build_panel_indices(state, panel):
    """Single-index colormap view: index picker + colorbar legend + value readout."""
    w = state.widgets
    _panel_title(panel, "🌿  Índices de vegetación")

    if not state.data_image:
        _panel_hint(panel, "Abre una imagen para calcular los índices.")
        return

    _panel_hint(panel, "Elige un índice y pasa el cursor sobre el visor para leer "
                       "el valor del píxel.")

    names = list(INDEX_DEFS.keys())
    ctk.CTkOptionMenu(
        panel, values=names, width=260, command=lambda n: _aplicar_indice(state, n),
        fg_color=COLORS["primary"], button_color=COLORS["primary_d"],
        button_hover_color=COLORS["primary_d"],
    ).pack(padx=18, pady=(4, 8))

    desc = ctk.CTkLabel(panel, text="", wraplength=270, justify="left", anchor="w",
                        font=ctk.CTkFont(size=11), text_color=COLORS["muted"])
    desc.pack(fill="x", padx=18, pady=(0, 8))
    w["lbl_index_desc"] = desc

    # Colorbar legend (gradient + min/mid/max ticks).
    legend = tk.Frame(panel, bg=COLORS["panel"])
    legend.pack(padx=18, pady=4)
    cb = tk.Label(legend, bg=COLORS["panel"])
    cb.pack(side="left")
    w["lbl_colorbar"] = cb
    ticks = tk.Frame(legend, bg=COLORS["panel"], height=170, width=54)
    ticks.pack(side="left", padx=8, fill="y")
    ticks.pack_propagate(False)
    w["lbl_cb_max"] = ctk.CTkLabel(ticks, text="", font=ctk.CTkFont(size=11),
                                   text_color=COLORS["text"])
    w["lbl_cb_max"].pack(side="top", anchor="w")
    w["lbl_cb_mid"] = ctk.CTkLabel(ticks, text="", font=ctk.CTkFont(size=11),
                                   text_color=COLORS["muted"])
    w["lbl_cb_mid"].pack(expand=True, anchor="w")
    w["lbl_cb_min"] = ctk.CTkLabel(ticks, text="", font=ctk.CTkFont(size=11),
                                   text_color=COLORS["text"])
    w["lbl_cb_min"].pack(side="bottom", anchor="w")

    # Per-pixel value readout.
    val = ctk.CTkLabel(panel, text="—",
                       font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
                       text_color=COLORS["primary"])
    val.pack(pady=(14, 0))
    w["lbl_index_value"] = val
    ctk.CTkLabel(panel, text="valor del píxel bajo el cursor",
                 font=ctk.CTkFont(size=10), text_color=COLORS["muted"]).pack()

    _aplicar_indice(state, names[0])


def _aplicar_indice(state, name):
    """Render the chosen index and refresh the colorbar legend + labels."""
    w = state.widgets
    mostrar_indice(state, name)
    d = INDEX_DEFS[name]

    if "lbl_index_desc" in w:
        w["lbl_index_desc"].configure(text=d["desc"])
    if "lbl_colorbar" in w:
        img = ImageTk.PhotoImage(colorbar_image(name))
        w["lbl_colorbar"].configure(image=img)
        w["lbl_colorbar"].image = img
    for key, v in (("lbl_cb_max", d["vmax"]),
                   ("lbl_cb_mid", (d["vmin"] + d["vmax"]) / 2),
                   ("lbl_cb_min", d["vmin"])):
        if key in w:
            w[key].configure(text=f"{v:.2f}")
    if "lbl_index_value" in w:
        w["lbl_index_value"].configure(text=f"{name} = —")


def _build_panel_segmentacion(state, panel):
    w = state.widgets
    _panel_title(panel, "✂  Segmentación")

    ctk.CTkLabel(panel, text="Manual", anchor="w",
                 font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                 text_color=COLORS["text"]).pack(fill="x", padx=18, pady=(4, 2))
    _panel_hint(panel, "Pulsa «Segmentación manual», dibuja el contorno arrastrando "
                       "el cursor sobre la imagen y crea la máscara.")
    _panel_button(panel, "✏  Segmentación manual", lambda: _activar_roi_manual(state))
    _panel_button(panel, "Crear máscara", lambda: mascara(state))

    tk.Frame(panel, bg="#e3e7ea", height=1).pack(fill="x", padx=18, pady=10)

    ctk.CTkLabel(panel, text="SAM", anchor="w",
                 font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                 text_color=COLORS["text"]).pack(fill="x", padx=18, pady=(4, 2))
    _panel_hint(panel, "1) Toma puntos y haz clic sobre el ROI (uno o varios).  "
                       "2) Segmenta.  3) Recorre las 3 máscaras con ◄ ► y deja "
                       "visible la que quieras usar para clasificar.")
    _panel_button(panel, "1 · Tomar puntos", lambda: sam_segm_flag(state))
    _panel_button(panel, "2 · Segmentar SAM", lambda: sam_segm(state))

    lbl_img_selec_seg = ctk.CTkLabel(panel, text="", text_color=COLORS["muted"],
                                     wraplength=270,
                                     font=ctk.CTkFont(family="Segoe UI", size=12))
    lbl_img_selec_seg.pack(padx=18, pady=(8, 2))
    w["lbl_img_selec_seg"] = lbl_img_selec_seg

    nav = tk.Frame(panel, bg=COLORS["panel"])
    nav.pack(padx=18, pady=4)
    ctk.CTkButton(nav, text="◄ Máscara", width=110, command=lambda: anterior_seg(state),
                  fg_color=COLORS["primary"], hover_color=COLORS["primary_d"]).pack(side="left", padx=4)
    ctk.CTkButton(nav, text="Máscara ►", width=110, command=lambda: siguiente_seg(state),
                  fg_color=COLORS["primary"], hover_color=COLORS["primary_d"]).pack(side="left", padx=4)


def _build_panel_clasificacion(state, panel):
    w = state.widgets
    _panel_title(panel, "🧠  Clasificación")
    _panel_hint(panel, "Segmenta un ROI (manual o SAM) y clasifícalo. El resultado "
                       "se muestra aquí con el nivel de confianza por clase.")
    _panel_button(panel, "Clasificar con AlexNet", lambda: _clasificar(state, "AlexNet"))
    _panel_button(panel, "Clasificar con AlgarroboNet", lambda: _clasificar(state, "AlgarroboNet"))

    card = ctk.CTkFrame(panel, fg_color="#eef2f4", corner_radius=12)
    card.pack(fill="x", padx=18, pady=16)
    w["_clasif_card"] = card
    ctk.CTkLabel(card, text="Sin resultado todavía", text_color=COLORS["muted"],
                 font=ctk.CTkFont(size=12)).pack(pady=22)


def _clasificar(state, modelo):
    """Run the chosen classifier and render the result card (no popup)."""
    if not state.save_mask and state.imagen_segementada is None:
        messagebox.showinfo("Clasificación", "Primero segmenta un ROI (manual o SAM).")
        return
    if modelo == "AlexNet":
        result = clasificar_alexnet_actualizado(state)
    else:
        result = clasificar_algarrobonet_actualizado(state)
    if result is not None:
        _render_clasif_card(state, result)


def _render_clasif_card(state, r):
    card = state.widgets.get("_clasif_card")
    if card is None:
        return
    for child in card.winfo_children():
        child.destroy()

    ctk.CTkLabel(card, text=f"Modelo · {r['model']}",
                 font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                 text_color=COLORS["muted"]).pack(anchor="w", padx=16, pady=(14, 0))

    clase = r["clase"]
    color = "#15803d" if clase == "PLUS" else "#b91c1c"
    ctk.CTkLabel(card, text=clase,
                 font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
                 text_color=color).pack(anchor="w", padx=16, pady=(2, 10))

    for name, p in r["probs"].items():
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(row, text=name, width=70, anchor="w",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text"]).pack(side="left")
        bar = ctk.CTkProgressBar(row, height=12, progress_color=COLORS["primary"])
        bar.set(float(p))
        bar.pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkLabel(row, text=f"{p * 100:.1f}%", width=52, anchor="e",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")

    ctk.CTkLabel(card, text="", height=6, fg_color="transparent").pack()


def _build_panel_correccion(state, panel):
    _panel_title(panel, "🎯  Corrección de bandas")
    _panel_hint(panel, "Activa la selección de puntos y marca con clic derecho el "
                       "mismo punto de referencia en cada banda; luego corrige para "
                       "alinearlas.")
    _panel_button(panel, "Activar selección de puntos",
                  lambda: _activar_correccion(state))
    _panel_button(panel, "Corregir bandas", lambda: correccion_imagen(state))


def _build_panel_entrenamiento(state, panel):
    _panel_title(panel, "⚙  Entrenamiento")
    _panel_button(panel, "Parámetros de red", ventana_configurar_parametros)
    _panel_button(panel, "Actualizar Database", lambda: _guardar_img(state))
    tk.Frame(panel, bg="#e3e7ea", height=1).pack(fill="x", padx=18, pady=10)
    _panel_button(panel, "Entrenar AlexNet", entrenar_modelo_alexnet)
    _panel_button(panel, "Entrenar AlgarroboNet", entrenar_algarrobonet)


# ── Helper actions ────────────────────────────────────────────────────────────

def _nav_band(state, fn):
    """Change band only on tools that allow it (Bandas / Corrección)."""
    if not state.data_image:
        return
    if state.widgets.get("_active_tool") not in ("bandas", "correccion"):
        return
    fn(state)


def _activar_roi_manual(state):
    """Enter manual ROI drawing mode and tell the user to trace the contour."""
    if not state.data_image:
        messagebox.showinfo("Segmentación", "Adjunte una imagen primero.")
        return
    state.flag_sam = False
    state.flag_roi = True
    state.contorno = []
    state.puntos_visibles = state.data_image[0].copy()
    lbl = state.widgets.get("lbl_img_selec_seg")
    if lbl is not None:
        lbl.configure(text="Dibujando contorno… luego pulsa «Crear máscara».")
    messagebox.showinfo(
        "Segmentación manual",
        "Dibuje el contorno del objeto de interés que desea segmentar, "
        "arrastrando el cursor sobre la imagen. Después pulse «Crear máscara».")


def _activar_correccion(state):
    state.flag_correccion = 1
    state.widgets["ttk_label_imagen"].configure(cursor="plus")
    messagebox.showinfo("Corrección", "Marque con clic derecho el punto de "
                                       "referencia en cada banda.")


def _abrir_mapa(state):
    """Centre the embedded map on the GPS coordinates of the loaded image."""
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


def _mostrar_ayuda(state):
    """Open a small, scrollable help window with the end-user usage guide."""
    parent = state.widgets["_ventana"]

    # Reuse the window if it is already open.
    existing = state.widgets.get("_ayuda_win")
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        existing.focus_set()
        return

    win = tk.Toplevel(parent)
    win.title("Ayuda — Guía de uso")
    win.configure(bg=COLORS["bg"])
    try:
        win.iconbitmap(str(Settings.APP_ICON))
    except Exception:
        pass

    ww, wh = 600, 700
    wh = min(wh, parent.winfo_screenheight() - 80)
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - ww) // 2)
    y = parent.winfo_rooty() + max(20, (parent.winfo_height() - wh) // 2)
    win.geometry(f"{ww}x{wh}+{x}+{y}")
    win.minsize(460, 420)
    win.transient(parent)
    state.widgets["_ayuda_win"] = win

    # Header band.
    header = tk.Frame(win, bg=COLORS["primary"], height=62)
    header.pack(fill="x")
    header.pack_propagate(False)
    ctk.CTkLabel(header, text="📖  Guía de uso", fg_color=COLORS["primary"],
                 font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
                 text_color="white").pack(side="left", padx=20)

    # Scrollable body with one card per section.
    body = ctk.CTkScrollableFrame(win, fg_color=COLORS["bg"])
    body.pack(fill="both", expand=True, padx=6, pady=6)

    for emoji, title, lines in HELP_SECTIONS:
        card = ctk.CTkFrame(body, fg_color="#ffffff", corner_radius=12)
        card.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(card, text=f"{emoji}  {title}", anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
                     text_color=COLORS["primary"]).pack(fill="x", padx=14, pady=(12, 6))
        for ln in lines:
            row = tk.Frame(card, bg="#ffffff")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(row, text="•", fg_color="#ffffff", text_color=COLORS["primary"],
                         font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", anchor="n")
            ctk.CTkLabel(row, text=ln, fg_color="#ffffff", justify="left", anchor="w",
                         wraplength=ww - 90, font=ctk.CTkFont(size=12),
                         text_color=COLORS["text"]).pack(side="left", padx=(6, 0), fill="x")
        ctk.CTkLabel(card, text="", height=2, fg_color="#ffffff").pack()

    # Footer with a close button.
    footer = tk.Frame(win, bg=COLORS["bg"])
    footer.pack(fill="x", pady=(2, 10))
    ctk.CTkButton(footer, text="Cerrar", width=140, command=win.destroy,
                  fg_color=COLORS["primary"], hover_color=COLORS["primary_d"]).pack()

    win.focus_set()


def _set_placeholder(label, array):
    img = ImageTk.PhotoImage(image=Image.fromarray(array))
    label.configure(image=img)
    label.image = img


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
