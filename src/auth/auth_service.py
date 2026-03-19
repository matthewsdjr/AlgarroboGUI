"""Authentication service: user registration, login and password hashing."""

import hashlib
import json
import os
import re
import tkinter as tk

from tkinter import messagebox
from PIL import Image, ImageTk

from config.settings import Settings


def cargar_usuarios() -> dict:
    path = str(Settings.USUARIOS_FILE)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def guardar_usuarios(usuarios: dict):
    with open(str(Settings.USUARIOS_FILE), "w") as f:
        json.dump(usuarios, f, indent=4)


def encriptar_contraseña(contraseña: str) -> str:
    return hashlib.sha256(contraseña.encode()).hexdigest()


def es_correo_valido(correo: str) -> bool:
    regex = r"^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return bool(re.match(regex, correo))


def es_alfabetico(cadena: str) -> bool:
    return bool(re.match(r"^[A-Za-záéíóúÁÉÍÓÚñÑ\s]+$", cadena))


def autenticar_usuario(usuario: str, contraseña: str) -> bool:
    """Return True and show welcome message when credentials are valid."""
    usuarios = cargar_usuarios()
    contraseña_encriptada = encriptar_contraseña(contraseña)
    if usuario in usuarios and usuarios[usuario]["contrasena"] == contraseña_encriptada:
        nombre = usuarios[usuario]["nombre"]
        apellido = usuarios[usuario]["apellido"]
        messagebox.showinfo("Éxito", f"Bienvenido {nombre} {apellido}!")
        return True
    messagebox.showerror("Error", "Usuario o contraseña incorrectos.")
    return False


def registrar_usuario(
    nombre: str,
    apellido: str,
    correo: str,
    nuevo_usuario: str,
    nueva_contraseña: str,
) -> bool:
    usuarios = cargar_usuarios()

    if not all([nombre, apellido, correo, nuevo_usuario, nueva_contraseña]):
        messagebox.showerror("Error", "Por favor, completa todos los campos.")
        return False
    if not es_correo_valido(correo):
        messagebox.showerror("Error", "Correo electrónico inválido.")
        return False
    if not es_alfabetico(nombre):
        messagebox.showerror("Error", "El nombre solo debe contener letras.")
        return False
    if not es_alfabetico(apellido):
        messagebox.showerror("Error", "El apellido solo debe contener letras.")
        return False

    for user, datos in usuarios.items():
        if user == nuevo_usuario:
            messagebox.showerror("Error", "El nombre de usuario ya existe.")
            return False
        if datos["correo"] == correo:
            messagebox.showerror("Error", "El correo electrónico ya está registrado.")
            return False

    usuarios[nuevo_usuario] = {
        "nombre": nombre,
        "apellido": apellido,
        "correo": correo,
        "contrasena": encriptar_contraseña(nueva_contraseña),
    }
    guardar_usuarios(usuarios)
    messagebox.showinfo("Éxito", "Usuario registrado exitosamente.")
    return True


def abrir_registro():
    """Open the user registration window."""
    registro_ventana = tk.Toplevel(bg="white")
    registro_ventana.title("Registro de Usuario")
    registro_ventana.geometry("380x450")

    try:
        img_user = Image.open(str(Settings.ICON_REGISTRAR)).resize((150, 150))
        foto_user = ImageTk.PhotoImage(img_user)
        label_foto = tk.Label(registro_ventana, image=foto_user, bg="white")
        label_foto.image = foto_user
        label_foto.grid(row=0, column=0, columnspan=2)
    except Exception:
        pass

    tk.Label(registro_ventana, text="Nombre:", font=("Arial", 15), fg="#4f4e4d", bg="white").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    nombre_entry = tk.Entry(registro_ventana, width=20, font=("Arial", 15), bg="white", highlightthickness=0, relief="flat", fg="#6b6a69")
    nombre_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Canvas(registro_ventana, width=180, height=2.0, bg="#bdb9b1", highlightthickness=0).grid(row=1, column=1, sticky="s", padx=(0, 30))

    tk.Label(registro_ventana, text="Apellido:", font=("Arial", 15), fg="#4f4e4d", bg="white").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    apellido_entry = tk.Entry(registro_ventana, width=20, font=("Arial", 15), bg="white", highlightthickness=0, relief="flat", fg="#6b6a69")
    apellido_entry.grid(row=2, column=1, padx=10, pady=5)
    tk.Canvas(registro_ventana, width=180, height=2.0, bg="#bdb9b1", highlightthickness=0).grid(row=2, column=1, sticky="s", padx=(0, 30))

    tk.Label(registro_ventana, text="E-mail:", font=("Arial", 15), fg="#4f4e4d", bg="white").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    correo_entry = tk.Entry(registro_ventana, width=20, font=("Arial", 15), bg="white", highlightthickness=0, relief="flat", fg="#6b6a69")
    correo_entry.grid(row=3, column=1, padx=10, pady=5)
    tk.Canvas(registro_ventana, width=180, height=2.0, bg="#bdb9b1", highlightthickness=0).grid(row=3, column=1, sticky="s", padx=(0, 30))

    tk.Label(registro_ventana, text="Usuario:", font=("Arial", 15), fg="#4f4e4d", bg="white").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    nuevo_usuario_entry = tk.Entry(registro_ventana, width=20, font=("Arial", 15), bg="white", highlightthickness=0, relief="flat", fg="#6b6a69")
    nuevo_usuario_entry.grid(row=4, column=1, padx=10, pady=5)
    tk.Canvas(registro_ventana, width=180, height=2.0, bg="#bdb9b1", highlightthickness=0).grid(row=4, column=1, sticky="s", padx=(0, 30))

    tk.Label(registro_ventana, text="Contraseña:", font=("Arial", 15), fg="#4f4e4d", bg="white").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    nueva_contra_entry = tk.Entry(registro_ventana, show="*", width=20, font=("Arial", 15), bg="white", highlightthickness=0, relief="flat", fg="#6b6a69")
    nueva_contra_entry.grid(row=5, column=1, padx=10, pady=5)
    tk.Canvas(registro_ventana, width=180, height=2.0, bg="#bdb9b1", highlightthickness=0).grid(row=5, column=1, sticky="s", padx=(0, 30))

    def _registrar():
        registrar_usuario(
            nombre_entry.get().strip(),
            apellido_entry.get().strip(),
            correo_entry.get().strip(),
            nuevo_usuario_entry.get().strip(),
            nueva_contra_entry.get().strip(),
        )
        registro_ventana.destroy()

    tk.Button(registro_ventana, text="Registrar", command=_registrar, font=("Arial", 15), bg="#6b6a69", fg="white", relief="flat").grid(row=6, column=0, columnspan=2, pady=30)
