"""AlgarroboNet CNN architecture and classification."""

import numpy as np
import cv2
import torch
import torch.nn as nn
from tkinter import messagebox
from torchvision import transforms

from config.settings import Settings


class CNNModel(nn.Module):
    """AlgarroboNet: lightweight CNN for Prosopis pallida classification."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=90, kernel_size=3, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.batchnorm = nn.BatchNorm2d(num_features=90, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=4556250, out_features=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def clasificar_algarrobonet_onnx(state):
    """Classify using the ONNX AlgarroboNet model."""
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

    entrada = state.AlgarroboNet.get_inputs()[0].name
    resultado = state.AlgarroboNet.run(None, {entrada: resized})
    etiqueta = np.argmax(resultado[0])

    clase = "Plus" if etiqueta == 1 else "No Plus"
    messagebox.showinfo("Clasificación AlgarroboNet", f"Clase Algarrobo: {clase}")


def clasificar_algarrobonet_actualizado(state):
    """Classify using a fine-tuned AlgarroboNet PyTorch checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNModel().to(device)
    model.load_state_dict(torch.load(str(Settings.ALGARROBONET_PTH)))
    model.to(device)
    model.eval()

    img = _get_segmented_image(state)
    if img is None:
        return

    resized = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
    preprocess = transforms.Compose([transforms.ToTensor()])
    tensor = preprocess(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["NO PLUS", "PLUS"]
    predicted_class = class_names[predicted.item()]
    messagebox.showinfo("AlgarroboNet", f"El Algarrobo pertenece a la clase: {predicted_class}")


def _get_segmented_image(state):
    if state.save_mask:
        return state.save_mask[state.contar_mask]["mask"]
    elif state.imagen_segementada is not None:
        return state.imagen_segementada
    else:
        messagebox.showerror("Error", "No hay imagen segmentada para clasificar.")
        return None
