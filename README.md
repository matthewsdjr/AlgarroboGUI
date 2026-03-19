# AlgarroboInterfaz

Interfaz gráfica para el análisis de imágenes multiespectrales de dron, segmentación con SAM (Segment Anything Model) y clasificación de árboles de Algarrobo (*Prosopis pallida*) mediante redes neuronales convolucionales.

## Estructura del Proyecto

```
AlgarroboInterfaz/
├── main.py                           # Punto de entrada de la aplicación
├── requirements.txt                  # Dependencias de Python
├── Dockerfile                        # Contenedor Docker
├── .gitignore
│
├── config/                           # Configuración centralizada
│   ├── settings.py                   # Rutas y constantes globales
│   ├── usuarios.json                 # Base de datos de usuarios
│   └── parametros.json               # Hiperparámetros de entrenamiento
│
├── assets/                           # Recursos gráficos
│   ├── icons/                        # Iconos de la interfaz
│   ├── logos/                        # Logos institucionales
│   ├── animations/                   # GIFs de carga
│   └── logo.ico                      # Icono de la aplicación
│
├── models/                           # Modelos de ML (no se suben a git)
│   ├── sam/                          # SAM (sam_vit_b_01ec64.pth)
│   ├── alexnet/                      # AlexNet (.onnx y .pth)
│   └── algarrobonet/                 # AlgarroboNet (.onnx y .pth)
│
├── data/                             # Datos de entrenamiento (no se suben a git)
│   └── DataStore227/                 # Imágenes de entrenamiento por combinación
│
└── src/                              # Código fuente modular
    ├── app_state.py                  # Estado global de la aplicación
    ├── auth/                         # Autenticación de usuarios
    │   └── auth_service.py
    ├── ui/                           # Ventanas de la interfaz
    │   ├── login_window.py           # Ventana de inicio de sesión
    │   └── main_window.py            # Ventana principal
    ├── image_processing/             # Procesamiento de imágenes
    │   ├── image_loader.py           # Carga de imágenes multiespectrales
    │   ├── image_viewer.py           # Visualización, zoom, navegación
    │   ├── spectral_indices.py       # Cálculo de índices vegetativos
    │   └── geometric_correction.py   # Corrección geométrica de bandas
    ├── segmentation/                 # Segmentación de copas de árboles
    │   ├── manual_segmentation.py    # Segmentación manual (ROI)
    │   └── sam_segmentation.py       # Segmentación con SAM
    ├── classification/               # Clasificación de árboles
    │   ├── model_loader.py           # Carga de modelos SAM y ONNX
    │   ├── alexnet_classifier.py     # Clasificación con AlexNet
    │   └── algarrobonet_classifier.py # Clasificación con AlgarroboNet
    └── training/                     # Entrenamiento de modelos
        ├── data_loader.py            # Carga de datasets
        ├── params_manager.py         # Gestión de hiperparámetros
        ├── trainer_alexnet.py        # Entrenamiento AlexNet
        └── trainer_algarrobonet.py   # Entrenamiento AlgarroboNet
```

## Requisitos Previos

- Python 3.10+
- CUDA (GPU NVIDIA) recomendado para SAM y entrenamiento
- Modelos descargados en `models/`

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd AlgarroboInterfaz

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración de Modelos

Descargar y colocar los modelos en las carpetas correspondientes:

| Modelo | Archivo | Carpeta destino |
|--------|---------|-----------------|
| SAM ViT-B | `sam_vit_b_01ec64.pth` | `models/sam/` |
| AlexNet ONNX | `netTransfer_alexnet.onnx` | `models/alexnet/` |
| AlexNet PyTorch | `alexnet_model_final.pth` | `models/alexnet/` |
| AlgarroboNet ONNX | `netTransfer_algarroboNet.onnx` | `models/algarrobonet/` |
| AlgarroboNet PyTorch | `cnn_model.pth` | `models/algarrobonet/` |

## Configuración de Assets

Colocar las imágenes de la interfaz en las carpetas correspondientes dentro de `assets/`:

- **`assets/icons/`**: `imagen.png`, `guardar.png`, `pregunta.png`, `map.png`, `copa.png`, `validacion.png`, `machine-learning.png`, `lapiz.png`, `aumentar.png`, `disminuir.png`, `siguiente.png`, `atras.png`, `tijeras.png`, `registrati-blu.png`
- **`assets/logos/`**: `algarrobo.jpg`, `UNF-844x1024_logo.png`, `PROCIENCIA-LOGO.png`, `PROCIENCIA.jpg`, `unf_fondo.png`
- **`assets/animations/`**: `abc.gif`
- **`assets/`**: `logo.ico`

## Ejecución

```bash
python main.py
```

## Docker

```bash
# Construir imagen
docker build -t algarrobo-interfaz .

# Ejecutar (requiere X11 forwarding para GUI)
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix algarrobo-interfaz
```

## Funcionalidades

1. **Login**: Sistema de autenticación con registro de usuarios
2. **Visualización**: Imágenes multiespectrales con 6 bandas (R, G, B, NIR, RE, RGB)
3. **Índices Vegetativos**: 24 índices (NDVI, EVI, ExG, NGRDI, etc.)
4. **Corrección Geométrica**: Alineación de bandas espectrales
5. **Segmentación Manual**: Trazado libre de ROI
6. **Segmentación SAM**: Segmentación por puntos con Segment Anything Model
7. **Clasificación**: AlexNet y AlgarroboNet para clasificar árboles Plus/No Plus
8. **Entrenamiento**: Interfaz para entrenar modelos con visualización en tiempo real
9. **Mapa**: Visualización georreferenciada con Google Satellite tiles
