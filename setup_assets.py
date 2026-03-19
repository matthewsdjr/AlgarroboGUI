"""
Script de configuración inicial.
Copia los assets (iconos, logos, modelos) desde la ubicación original
a la estructura organizada del proyecto.

Ejecutar una sola vez:  python setup_assets.py
"""

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE_S = Path("S:/SOFT. ALGARROBO MATT")
SOURCE_LOCAL = Path("C:/Users/INOC_2/Documents/SOFT. ALGARROBO MATT")
SOURCE_DESKTOP = Path("C:/Users/INOC_2/OneDrive/Escritorio/funciones_cargar")
SOURCE_MODELS_ALG = Path("C:/Users/INOC_2/OneDrive/Escritorio/Entrenamiento/modelo algarrobonet")
SOURCE_MODELS_ALEX = Path("C:/Users/INOC_2/OneDrive/Escritorio/Entrenamiento/modelo_alexnet")


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        print(f"  OK: {src.name} -> {dst}")
    else:
        print(f"  FALTA: {src}")


def main():
    print("=== Copiando iconos ===")
    icons = [
        "imagen.png", "guardar.png", "pregunta.png", "map.png",
        "copa.png", "validacion.png", "machine-learning.png",
        "lapiz.png", "aumentar.png", "disminuir.png",
        "siguiente.png", "atras.png", "tijeras.png", "registrati-blu.png",
        "data.png",
    ]
    for icon in icons:
        for src_dir in [SOURCE_S, SOURCE_LOCAL]:
            src = src_dir / icon
            if src.exists():
                copy_if_exists(src, BASE_DIR / "assets" / "icons" / icon)
                break
        else:
            print(f"  NO ENCONTRADO: {icon}")

    print("\n=== Copiando logos ===")
    logos = {
        "algarrobo.jpg": "algarrobo.jpg",
        "UNF-844x1024_logo.png": "UNF-844x1024_logo.png",
        "PROCIENCIA-LOGO.png": "PROCIENCIA-LOGO.png",
        "PROCIENCIA.jpg": "PROCIENCIA.jpg",
        "unf_fondo.png": "unf_fondo.png",
    }
    for src_name, dst_name in logos.items():
        for src_dir in [SOURCE_S, SOURCE_LOCAL]:
            src = src_dir / src_name
            if src.exists():
                copy_if_exists(src, BASE_DIR / "assets" / "logos" / dst_name)
                break
        else:
            print(f"  NO ENCONTRADO: {src_name}")

    print("\n=== Copiando animaciones ===")
    for src_dir in [SOURCE_S, SOURCE_LOCAL]:
        src = src_dir / "abc.gif"
        if src.exists():
            copy_if_exists(src, BASE_DIR / "assets" / "animations" / "abc.gif")
            break

    print("\n=== Copiando icono de app ===")
    for src_dir in [SOURCE_S, SOURCE_LOCAL]:
        src = src_dir / "logo.ico"
        if src.exists():
            copy_if_exists(src, BASE_DIR / "assets" / "logo.ico")
            break

    print("\n=== Copiando modelos ===")
    model_files = [
        (SOURCE_DESKTOP / "sam_vit_b_01ec64.pth", BASE_DIR / "models" / "sam" / "sam_vit_b_01ec64.pth"),
        (SOURCE_S / "sam_vit_b_01ec64.pth", BASE_DIR / "models" / "sam" / "sam_vit_b_01ec64.pth"),
        (SOURCE_DESKTOP / "netTransfer_alexnet.onnx", BASE_DIR / "models" / "alexnet" / "netTransfer_alexnet.onnx"),
        (SOURCE_DESKTOP / "netTransfer_algarroboNet.onnx", BASE_DIR / "models" / "algarrobonet" / "netTransfer_algarroboNet.onnx"),
        (SOURCE_MODELS_ALEX / "alexnet_model_final.pth", BASE_DIR / "models" / "alexnet" / "alexnet_model_final.pth"),
        (SOURCE_MODELS_ALG / "cnn_model.pth", BASE_DIR / "models" / "algarrobonet" / "cnn_model.pth"),
    ]
    copied = set()
    for src, dst in model_files:
        if dst.name not in copied:
            copy_if_exists(src, dst)
            if src.exists():
                copied.add(dst.name)

    print("\n=== Configuración completada ===")
    print("Revisa los mensajes 'FALTA' y copia manualmente los archivos faltantes.")


if __name__ == "__main__":
    main()
