"""Manage training hyper-parameters stored in parametros.json."""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

from config.settings import Settings


def cargar_parametros() -> dict:
    path = str(Settings.PARAMETROS_FILE)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def guardar_parametros(parametros: dict):
    with open(str(Settings.PARAMETROS_FILE), "w") as f:
        json.dump(parametros, f, indent=3)


def ventana_configurar_parametros():
    """Open a window to view / edit training hyper-parameters."""

    conf_red = tk.Toplevel(bg="white")
    conf_red.title("Parámetros de Red")
    conf_red.geometry("360x170")

    tk.Label(conf_red, text="Learning Rate", bg="white").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
    tk.Label(conf_red, text="Batch", bg="white").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
    tk.Label(conf_red, text="Epochs", bg="white").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

    entry_lr = tk.Entry(conf_red, width=20, bd=3)
    entry_lr.grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
    entry_batch = tk.Entry(conf_red, width=20, bd=3)
    entry_batch.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)
    entry_epoch = tk.Entry(conf_red, width=20, bd=3)
    entry_epoch.grid(column=1, row=2, sticky=tk.NS, padx=5, pady=5)

    tk.Label(conf_red, text="Valor entre 0.001 - 0.01", bg="white").grid(column=2, row=0, padx=5, pady=5)
    tk.Label(conf_red, text="Valor entre 20 - 120", bg="white").grid(column=2, row=1, padx=5, pady=5)
    tk.Label(conf_red, text="Valor entre 1 - 200", bg="white").grid(column=2, row=2, padx=5, pady=5)

    check_alex = tk.BooleanVar()
    check_alg = tk.BooleanVar()

    def cargar_valores_red(red):
        entry_lr.delete(0, tk.END)
        entry_batch.delete(0, tk.END)
        entry_epoch.delete(0, tk.END)
        if red == "AlexNet":
            check_alg.set(False)
        else:
            check_alex.set(False)
        parametros = cargar_parametros()
        if red in parametros:
            entry_lr.insert(0, parametros[red]["lr"])
            entry_batch.insert(0, parametros[red]["batch"])
            entry_epoch.insert(0, parametros[red]["epochs"])

    ttk.Checkbutton(conf_red, text="AlexNet", variable=check_alex,
                     command=lambda: cargar_valores_red("AlexNet")).grid(column=0, row=3, padx=5, pady=5)
    ttk.Checkbutton(conf_red, text="AlgarroboNet", variable=check_alg,
                     command=lambda: cargar_valores_red("AlgarroboNet")).grid(column=1, row=3, padx=5, pady=5)

    def actualizar():
        parametros = cargar_parametros()
        if check_alex.get() != check_alg.get():
            key = "AlexNet" if check_alex.get() else "AlgarroboNet"
            parametros[key] = {
                "lr": entry_lr.get(),
                "batch": entry_batch.get(),
                "epochs": entry_epoch.get(),
            }
            guardar_parametros(parametros)
            messagebox.showinfo("OK", f"Parámetros de {key} actualizados.")
        else:
            messagebox.showerror("Error", "Seleccione una de las redes.")

    tk.Button(conf_red, text="Actualizar parámetros", command=actualizar).grid(
        column=1, row=4, sticky=tk.NSEW, padx=5, pady=5
    )

    check_alg.set(True)
    cargar_valores_red("AlgarroboNet")
