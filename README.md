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

## Interfaz de Usuario (Rediseño v2)

La interfaz se rediseñó por completo pasando de una disposición de **dos frames** a un
**visor único central**, manteniendo intacta toda la lógica de procesamiento.

- **Encabezado institucional fijo**: logo de la UNF a la izquierda, logo de Prociencia a la
  derecha y el título del proyecto centrado, siempre visible.
- **Barra de herramientas superior**: acceso a Bandas, Índices, Mapa, Segmentación,
  Clasificación, Corrección, Entrenamiento y Guardar. Cada herramienta se muestra **dentro
  del mismo visor**, sin abrir ventanas emergentes.
- **Visor único** con un **panel contextual** lateral que cambia según la herramienta activa.
- **Ventana redimensionable** y centrada al iniciar (ya no se abre bloqueada a pantalla
  completa); el usuario puede ajustar el tamaño o maximizarla.

### Navegación de bandas
- Se muestran **únicamente las bandas entregadas por el dron** (si una toma DJI no trae la
  banda azul, esta se oculta). Un contador indica la posición, p. ej. `RED · 2/6`.
- Navegación con **flechas en pantalla** (junto a la imagen) y con el **teclado (← / →)**.
  Las flechas aparecen solo en las pestañas donde tiene sentido cambiar de banda (Bandas y
  Corrección).
- **Zoom con la rueda del mouse** y **desplazamiento arrastrando** sobre la imagen.

### Índices de vegetación
- Selección de **un índice** que se renderiza con un **mapa de color** (rojo→verde) y una
  **leyenda / colorbar** con la escala de valores.
- **Lectura por píxel en tiempo real**: al pasar el cursor sobre la imagen se muestra el
  valor del índice en ese punto.
- Índices disponibles: **NDVI, GNDVI, NDRE, NGRDI, NGBDI, RVI, SCCI**.

### Segmentación
- **Manual**: se activa desde el panel, se traza el contorno arrastrando el cursor y se crea
  la máscara, que se muestra en el visor.
- **SAM**: se marcan **uno o varios puntos** sobre el ROI y, tras segmentar, las **3
  máscaras** candidatas se muestran en el visor para elegir con las flechas (◄ ►).

### Clasificación
- El resultado se muestra como una **tarjeta integrada** en el visor (sin ventana
  emergente): modelo usado (**AlexNet** / **AlgarroboNet**), clase predicha (Plus / No Plus)
  y **probabilidades por clase** con barras de confianza.

### Mapa
- Mapa **embebido** dentro del visor (Google Satellite tiles), centrado en las coordenadas
  GPS extraídas de la imagen, con un botón para volver a la vista de la imagen.

### Corrección geométrica
- Es **opcional y a iniciativa del usuario** desde la barra superior (se eliminó la consulta
  automática que aparecía al abrir el programa).

## Funcionalidades

1. **Login**: Sistema de autenticación con registro de usuarios.
2. **Visualización**: Imágenes multiespectrales (R, G, B, NIR, RE y compuesta RGB), mostrando
   solo las bandas disponibles.
3. **Índices Vegetativos**: vista de índice único con colormap, colorbar y valor por píxel
   (NDVI, GNDVI, NDRE, etc.). Internamente se calculan 24 índices para exportación `.mat`.
4. **Corrección Geométrica**: alineación de bandas espectrales (opcional).
5. **Segmentación Manual**: trazado libre de ROI sobre el visor.
6. **Segmentación SAM**: segmentación multipunto con Segment Anything Model y selección de
   máscara en el visor.
7. **Clasificación**: AlexNet y AlgarroboNet con tarjeta de resultado y probabilidades.
8. **Entrenamiento**: interfaz para entrenar modelos con visualización en tiempo real.
9. **Mapa**: visualización georreferenciada embebida con Google Satellite tiles.
