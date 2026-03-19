"""AlexNet-based tree classification (ONNX and PyTorch)."""

import numpy as np
import cv2
import torch
import torch.nn as nn
from tkinter import messagebox
from torchvision import models, transforms

from config.settings import Settings


def clasificar_algarrobo_alexnet_onnx(state):
    """Classify using the ONNX AlexNet model."""
    if state.save_mask:
        img = state.save_mask[state.contar_mask]["mask"]
    elif state.imagen_segementada is not None:
        img = state.imagen_segementada
    else:
        messagebox.showerror("Error", "No hay imagen segmentada para clasificar.")
        return

    resized = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32).transpose(2, 0, 1)
    resized = np.expand_dims(resized, axis=0)

    entrada = state.net.get_inputs()[0].name
    resultado = state.net.run(None, {entrada: resized})
    etiqueta = np.argmax(resultado[0])

    clase = "Plus" if etiqueta == 1 else "No Plus"
    messagebox.showinfo("Clasificación AlexNet", f"Clase Algarrobo: {clase}")


def clasificar_alexnet_actualizado(state):
    """Classify using a fine-tuned AlexNet PyTorch checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 2)
    alexnet.load_state_dict(torch.load(str(Settings.ALEXNET_PTH)))
    alexnet.to(device)
    alexnet.eval()

    img = _get_segmented_image(state)
    if img is None:
        return

    resized = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
    preprocess = transforms.Compose([transforms.ToTensor()])
    tensor = preprocess(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = alexnet(tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["NO PLUS", "PLUS"]
    predicted_class = class_names[predicted.item()]
    messagebox.showinfo("AlexNet", f"El Algarrobo pertenece a la clase: {predicted_class}")


def _get_segmented_image(state):
    if state.save_mask:
        return state.save_mask[state.contar_mask]["mask"]
    elif state.imagen_segementada is not None:
        return state.imagen_segementada
    else:
        messagebox.showerror("Error", "No hay imagen segmentada para clasificar.")
        return None
