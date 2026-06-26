# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AlgarroboInterfaz** is a desktop GUI application for multispectral drone image analysis and ML-based classification of *Prosopis pallida* (Algarrobo) trees. Users load drone survey images, apply vegetation indices, segment trees (manually or via SAM), classify them (Plus/No-Plus), and optionally fine-tune the CNN models.

## Running the Application

Activate the conda environment before running anything:

```bash
conda activate algarrobogui
```

Then launch the app:

```bash
python main.py
```

**Asset/model setup** (required on first run if models aren't present):
```bash
python setup_assets.py
```

**Docker** (requires X11 forwarding on Linux):
```bash
docker build -t algarrobo-interfaz .
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix algarrobo-interfaz
```

There is no test suite. Verification is done by running the application manually.

## Architecture

### Entry Point & State

`main.py` creates an `AppState` instance, builds the Tkinter root window, shows the login window, then transitions to the main application window. `src/app_state.py` acts as a central state container shared across all modules — it holds loaded images, model handles, current selections, and UI references. Pass or access it rather than using module-level globals.

### Module Map

| Package | Responsibility |
|---|---|
| `src/auth/` | Login validation and user registration against `config/usuarios.json` |
| `src/ui/` | Login window and main application window (tab layout) |
| `src/image_processing/` | Loading 6-band drone images (R, G, B, NIR, RE, RGB), computing 24 vegetation indices, geometric correction |
| `src/segmentation/` | Manual ROI polygon drawing and SAM (Segment Anything Model) auto-segmentation |
| `src/classification/` | Inference via AlexNet or AlgarroboNet using ONNX Runtime or PyTorch |
| `src/training/` | Fine-tuning pipeline, hyperparameter management from `config/parametros.json` |
| `config/settings.py` | All file paths and constants in one place — update paths here, not scattered in code |

### Data Flow

1. **Auth** — credentials checked against `config/usuarios.json`
2. **Load images** — drone survey folder → 6 spectral bands parsed from filenames
3. **Index computation** — selected vegetation index applied per-pixel → display canvas
4. **Segmentation** — polygon ROI drawn manually OR SAM model generates masks
5. **Classification** — cropped ROI patch → CNN → Plus / No-Plus label
6. **Training** — labeled ROIs in `data/IMAGENES/PLUS/` and `NO_PLUS/` → fine-tune model

### ML Models

Models live in `models/` (git-ignored, populated by `setup_assets.py`):
- `models/sam/` — SAM ViT-B checkpoint for segmentation
- `models/alexnet/` — AlexNet (`.onnx` + `.pt`)
- `models/algarrobonet/` — AlgarroboNet (`.onnx` + `.pt`)

ONNX Runtime is the default inference backend; PyTorch is used for training.

### Configuration

- `config/settings.py` — centralizes all directory and file paths
- `config/usuarios.json` — user credentials (plain JSON, not encrypted)
- `config/parametros.json` — training hyperparameters (epochs, LR, batch size, etc.)

### Supported Vegetation Indices

24 indices including NDVI, EVI, NDRE, ExG, CIVE, NGRDI, RGBVI, SCCI, and others. All are implemented in `src/image_processing/`.

## Key Conventions

- The GUI uses **customtkinter** widgets throughout — prefer `ctk.*` over standard `tk.*` for new UI elements to maintain visual consistency.
- The map panel uses **TkinterMapView** with Google Satellite tiles, tied to image EXIF GPS coordinates.
- Image bands are referenced by index (0=R, 1=G, 2=B, 3=NIR, 4=RE, 5=RGB composite) in processing code.
- All new source paths should be registered in `config/settings.py`, not hardcoded inline.
