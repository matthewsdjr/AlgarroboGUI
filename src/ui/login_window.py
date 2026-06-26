"""Login window: user authentication and registration entry point."""

import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk

from config.settings import Settings
from src.auth.auth_service import autenticar_usuario, abrir_registro


def build_login_window(root, on_login_success):
    """Create and populate the login window.

    Parameters
    ----------
    root : tk.Tk
        The root Tk window.
    on_login_success : callable
        Called (with no args) when authentication succeeds.
    """

    root.title("Login")
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    win_w = min(sw, 1535)
    win_h = min(sh, 850)
    root.geometry(f"{win_w}x{win_h}")
    root.resizable(False, False)

    # ── Background ───────────────────────────────────────────────
    fondo = Image.open(str(Settings.FONDO_LOGIN)).resize((win_w, win_h), Image.LANCZOS)
    foto = ImageTk.PhotoImage(fondo)
    panel = tk.Label(root, image=foto)
    panel.image = foto
    panel.pack(fill="both", expand="yes")

    # ── Login frame ──────────────────────────────────────────────
    log_frame = ctk.CTkFrame(
        root, fg_color="white", width=950, height=600,
        corner_radius=50, bg_color="transparent",
    )
    log_frame.place(relx=0.5, rely=0.5, anchor="center")

    # ── Algarrobo image ──────────────────────────────────────────
    alg_img = Image.open(str(Settings.LOGO_ALGARROBO)).resize((400, 400))
    foto_alg = ImageTk.PhotoImage(alg_img)
    lbl_alg = tk.Label(log_frame, image=foto_alg)
    lbl_alg.image = foto_alg
    lbl_alg.place(relx=0.05, rely=0.15)

    # ── UNF logo ─────────────────────────────────────────────────
    unf_img = Image.open(str(Settings.LOGO_UNF)).resize((115, 145))
    foto_unf = ImageTk.PhotoImage(unf_img)
    lbl_unf = tk.Label(log_frame, image=foto_unf, bg="white")
    lbl_unf.image = foto_unf
    lbl_unf.place(relx=0.54, rely=0.1)

    # ── Prociencia logo ──────────────────────────────────────────
    pro_img = Image.open(str(Settings.LOGO_PROCIENCIA)).resize((260, 145))
    foto_pro = ImageTk.PhotoImage(pro_img)
    lbl_pro = tk.Label(log_frame, image=foto_pro, bg="white")
    lbl_pro.image = foto_pro
    lbl_pro.place(relx=0.7, rely=0.07)

    # ── Username ─────────────────────────────────────────────────
    tk.Label(log_frame, text="Usuario", font=("Arial", 15), fg="#4f4e4d", bg="white").place(x=530, y=250)
    username_entry = tk.Entry(
        log_frame, width=20, font=("Arial", 15), bg="white",
        highlightthickness=0, relief="flat", fg="#6b6a69",
    )
    username_entry.place(x=530, y=285)
    tk.Canvas(log_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0).place(x=530, y=310)

    # ── Password ─────────────────────────────────────────────────
    tk.Label(log_frame, text="Contraseña", font=("Arial", 15), fg="#4f4e4d", bg="white").place(x=530, y=340)
    password_entry = tk.Entry(
        log_frame, width=20, show="*", font=("Arial", 15), bg="white",
        highlightthickness=0, relief="flat", fg="#6b6a69",
    )
    password_entry.place(x=530, y=375)
    tk.Canvas(log_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0).place(x=530, y=400)

    # ── Buttons ──────────────────────────────────────────────────
    def _do_login():
        usuario = username_entry.get().strip()
        contraseña = password_entry.get().strip()
        if autenticar_usuario(usuario, contraseña):
            on_login_success()

    username_entry.bind("<Return>", lambda e: _do_login())
    password_entry.bind("<Return>", lambda e: _do_login())

    tk.Button(
        log_frame, text="Iniciar Sesión", command=_do_login,
        font=("Arial", 15), bg="#6b6a69", fg="white", relief="flat",
    ).place(x=770, y=460, anchor="center")

    tk.Button(
        log_frame, text="Regístrate", command=abrir_registro,
        font=("Arial", 15), bg="#6b6a69", fg="white", relief="flat",
    ).place(x=600, y=460, anchor="center")
