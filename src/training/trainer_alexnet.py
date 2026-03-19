"""AlexNet training loop with live visualisation."""

import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.training.params_manager import cargar_parametros
from src.training.data_loader import cargar_database


def entrenar_modelo_alexnet():
    parametros = cargar_parametros()
    val_lr = float(parametros["AlexNet"]["lr"])
    val_batch = int(parametros["AlexNet"]["batch"])
    val_epochs = int(parametros["AlexNet"]["epochs"])

    device, train_dataset, val_dataset, num_classes = cargar_database()

    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier = nn.Sequential(
        nn.Dropout(), nn.Linear(9216, 4096), nn.ReLU(inplace=True),
        nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    alexnet = alexnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(), lr=val_lr, momentum=0.9)

    train_loader = DataLoader(train_dataset, batch_size=val_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True)

    vis_window = tk.Toplevel()
    vis_window.title("Progreso de Entrenamiento AlexNet")
    vis_window.geometry("900x700")

    main_frame = tk.Frame(vis_window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    plot_frame = tk.Frame(main_frame)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    metrics_frame = tk.Frame(main_frame, width=200, bg="white")
    metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    tk.Label(metrics_frame, text="Métricas", font=("Arial", 12, "bold"), bg="white").pack(pady=10)
    tk.Label(metrics_frame, text="Epoch:", font=("Arial", 10, "bold"), bg="white").pack(pady=5)
    lbl_epoch = tk.Label(metrics_frame, text="--", font=("Arial", 10), bg="white"); lbl_epoch.pack()
    tk.Label(metrics_frame, text="Loss:", font=("Arial", 10, "bold"), bg="white").pack(pady=5)
    lbl_loss = tk.Label(metrics_frame, text="--", font=("Arial", 10), bg="white"); lbl_loss.pack()
    tk.Label(metrics_frame, text="Train Accuracy:", font=("Arial", 10, "bold"), bg="white").pack(pady=5)
    lbl_train = tk.Label(metrics_frame, text="--", font=("Arial", 10), bg="white"); lbl_train.pack()
    tk.Label(metrics_frame, text="Val Accuracy:", font=("Arial", 10, "bold"), bg="white").pack(pady=5)
    lbl_val = tk.Label(metrics_frame, text="--", font=("Arial", 10), bg="white"); lbl_val.pack()

    epochs_list, losses, train_accs, val_accs = [], [], [], []

    def update_plots():
        ax1.clear(); ax2.clear()
        ax1.plot(epochs_list, losses, label="Training Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Training Loss")
        ax2.plot(epochs_list, train_accs, label="Training Accuracy")
        ax2.plot(epochs_list, val_accs, label="Validation Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.set_title("Accuracy")
        canvas.draw()

    final_val_acc = 0
    for epoch in range(val_epochs):
        alexnet.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0); correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        alexnet.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = alexnet(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0); val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        final_val_acc = val_acc

        epochs_list.append(epoch + 1); losses.append(epoch_loss)
        train_accs.append(epoch_acc); val_accs.append(val_acc)
        update_plots()

        lbl_epoch.config(text=str(epoch + 1))
        lbl_loss.config(text=f"{epoch_loss:.4f}")
        lbl_train.config(text=f"{epoch_acc:.2f}%")
        lbl_val.config(text=f"{val_acc:.2f}%")
        vis_window.update()

    messagebox.showinfo("Entrenamiento Completado", f"Precisión final: {final_val_acc:.2f}%")
    vis_window.destroy()
