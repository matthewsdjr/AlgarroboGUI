import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Centralised paths and constants for the whole application."""

    # ── Directories ──────────────────────────────────────────────
    ASSETS_DIR = BASE_DIR / "assets"
    ICONS_DIR = ASSETS_DIR / "icons"
    LOGOS_DIR = ASSETS_DIR / "logos"
    ANIMATIONS_DIR = ASSETS_DIR / "animations"

    MODELS_DIR = BASE_DIR / "models"
    SAM_DIR = MODELS_DIR / "sam"
    ALEXNET_DIR = MODELS_DIR / "alexnet"
    ALGARROBONET_DIR = MODELS_DIR / "algarrobonet"

    DATA_DIR = BASE_DIR / "data"
    CONFIG_DIR = BASE_DIR / "config"

    # ── Authentication ───────────────────────────────────────────
    USUARIOS_FILE = CONFIG_DIR / "usuarios.json"

    # ── Training parameters ──────────────────────────────────────
    PARAMETROS_FILE = CONFIG_DIR / "parametros.json"

    # ── Model files ──────────────────────────────────────────────
    SAM_CHECKPOINT = SAM_DIR / "sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"

    ALEXNET_ONNX = ALEXNET_DIR / "netTransfer_alexnet.onnx"
    ALEXNET_PTH = ALEXNET_DIR / "alexnet_model_final.pth"

    ALGARROBONET_ONNX = ALGARROBONET_DIR / "netTransfer_algarroboNet.onnx"
    ALGARROBONET_PTH = ALGARROBONET_DIR / "cnn_model.pth"

    # ── Asset files ──────────────────────────────────────────────
    APP_ICON = ASSETS_DIR / "logo.ico"
    LOADING_GIF = ANIMATIONS_DIR / "abc.gif"

    # Icons
    ICON_IMAGEN = ICONS_DIR / "imagen.png"
    ICON_DATA = ICONS_DIR / "data.png"
    ICON_GUARDAR = ICONS_DIR / "guardar.png"
    ICON_PREGUNTA = ICONS_DIR / "pregunta.png"
    ICON_MAP = ICONS_DIR / "map.png"
    ICON_COPA = ICONS_DIR / "copa.png"
    ICON_VALIDACION = ICONS_DIR / "validacion.png"
    ICON_ML = ICONS_DIR / "machine-learning.png"
    ICON_LAPIZ = ICONS_DIR / "lapiz.png"
    ICON_AUMENTAR = ICONS_DIR / "aumentar.png"
    ICON_DISMINUIR = ICONS_DIR / "disminuir.png"
    ICON_SIGUIENTE = ICONS_DIR / "siguiente.png"
    ICON_ATRAS = ICONS_DIR / "atras.png"
    ICON_TIJERAS = ICONS_DIR / "tijeras.png"
    ICON_REGISTRAR = ICONS_DIR / "registrati-blu.png"

    # Logos
    LOGO_ALGARROBO = LOGOS_DIR / "algarrobo.jpg"
    LOGO_UNF = LOGOS_DIR / "UNF-844x1024_logo.png"
    LOGO_PROCIENCIA = LOGOS_DIR / "PROCIENCIA-LOGO.png"
    LOGO_PROCIENCIA_JPG = LOGOS_DIR / "PROCIENCIA.jpg"
    FONDO_LOGIN = LOGOS_DIR / "unf_fondo.png"

    # ── Training data ────────────────────────────────────────────
    DATASTORE_DIR = DATA_DIR / "DataStore227"

    # ── Image display constants ──────────────────────────────────
    IMG_WIDTH = 1600
    IMG_HEIGHT = 1300
    DISPLAY_FACTOR = 0.5
    ZOOM_FACTOR_IN = 1.1
    ZOOM_FACTOR_OUT = 0.9

    # ── Vegetation index labels ──────────────────────────────────
    INDEX_LABELS = [
        "IMAGEN PRINCIPAL", "RED", "GREEN", "BLUE", "EDGE RED", "NIR",
        "EXG", "EXGR", "NGRDI", "NGBDI", "RGRI", "GBRI", "CIVE", "VEG",
        "RGBVI", "MGRVI", "NDVI", "RVI", "DVI", "EVI", "REVI", "NDRE",
        "REEVI", "REDVI", "SCCI",
    ]

    # ── Device ───────────────────────────────────────────────────
    DEVICE = "cuda"

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist."""
        for d in (
            cls.ASSETS_DIR, cls.ICONS_DIR, cls.LOGOS_DIR, cls.ANIMATIONS_DIR,
            cls.MODELS_DIR, cls.SAM_DIR, cls.ALEXNET_DIR, cls.ALGARROBONET_DIR,
            cls.DATA_DIR, cls.CONFIG_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)
