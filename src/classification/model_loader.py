"""Load ML models: SAM, ONNX classifiers and PyTorch checkpoints."""

import threading
import onnxruntime as ort
from segment_anything import sam_model_registry

from config.settings import Settings


def cargar_modelo(state):
    """Load SAM model and ONNX classification models into state."""
    sam_checkpoint = str(Settings.SAM_CHECKPOINT)
    model_type = Settings.SAM_MODEL_TYPE
    device = Settings.DEVICE

    state.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    state.sam.to(device=device)
    print("Modelo SAM cargado con éxito.")

    state.net = ort.InferenceSession(str(Settings.ALEXNET_ONNX))
    print("Modelo AlexNet ONNX cargado.")

    state.AlgarroboNet = ort.InferenceSession(str(Settings.ALGARROBONET_ONNX))
    print("Modelo AlgarroboNet ONNX cargado.")


def iniciar_carga_modelo(state):
    """Start model loading in a background thread."""
    threading.Thread(target=cargar_modelo, args=(state,), daemon=True).start()
