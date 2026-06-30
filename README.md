# AlgarroboInterfaz

Interfaz gráfica para el análisis de imágenes multiespectrales de dron, segmentación con SAM (Segment Anything Model) y clasificación de árboles de Algarrobo (*Prosopis pallida*) mediante redes neuronales convolucionales.

## Interfaz de Usuario

La aplicación se organiza en torno a un **visor único central** y está compuesta por:

- **Encabezado** fijo: logo de la UNF a la izquierda, logo de Prociencia a la derecha y el
  título del proyecto centrado, siempre visible.
- **Barra de herramientas** superior con acceso a: *Abrir imagen, Bandas, Índices, Mapa,
  Segmentación, Clasificación, Corrección, Entrenamiento* y *Guardar*. Cada herramienta se
  muestra dentro del mismo visor.
- **Visor central** donde se muestra la imagen, los índices, el mapa o las máscaras.
- **Panel lateral** contextual que cambia según la herramienta activa.
- **Pie de página** con la versión de la interfaz y los derechos institucionales.

La ventana se abre centrada y se puede redimensionar o maximizar libremente.

### Abrir una imagen

Pulsa **«Abrir imagen»** y selecciona la toma del dron. Las bandas se cargan
automáticamente y la vista se posiciona en la imagen principal (compuesta RGB).

### Navegación de bandas

- Se muestran **únicamente las bandas entregadas por el dron** (si una toma no trae la banda
  azul, esta no aparece). Un contador indica la posición actual, p. ej. `RED · 2/6`.
- Cambia de banda con las **flechas en pantalla** (junto a la imagen) o con el **teclado
  (← / →)**. Las flechas están disponibles en las pestañas **Bandas** y **Corrección**.
- **Zoom** con la rueda del mouse y **desplazamiento** arrastrando sobre la imagen.
- Al entrar a **Bandas** se muestra siempre la imagen principal.

### Índices de vegetación

1. Abre la pestaña **Índices**.
2. Elige un índice en el menú desplegable.
3. El índice se dibuja en el visor con un **mapa de color** (rojo→verde) y una
   **leyenda / colorbar** con la escala de valores.
4. Pasa el cursor sobre la imagen para ver, en tiempo real, el **valor del índice en ese
   píxel**.

Índices disponibles: **NDVI, GNDVI, NDRE, NGRDI, NGBDI, RVI, SCCI**.

### Segmentación

En la pestaña **Segmentación** se muestra la imagen principal. Hay dos modos:

- **Manual**: pulsa *«Segmentación manual»*, traza el contorno del objeto arrastrando el
  cursor sobre la imagen y pulsa *«Crear máscara»*. La máscara resultante se muestra en el
  visor.
- **SAM**: pulsa *«Tomar puntos»* y haz clic sobre el ROI (uno o varios puntos). Luego pulsa
  *«Segmentar SAM»*: se generan **3 máscaras** candidatas que puedes recorrer con las flechas
  (◄ ►) directamente en el visor. La máscara visible es la que se usará para clasificar.

### Clasificación

1. Segmenta primero un ROI (manual o SAM).
2. En la pestaña **Clasificación**, elige el modelo (**AlexNet** o **AlgarroboNet**).
3. El resultado aparece como una **tarjeta** en el panel: modelo usado, clase predicha
   (**Plus** / **No Plus**) y las **probabilidades por clase** con barras de confianza.

### Mapa

En la pestaña **Mapa** se muestra, dentro del visor, la **ubicación de la toma** sobre un
mapa satelital, centrado en las coordenadas GPS extraídas de la imagen. Un botón permite
**volver a la vista de la imagen**.

### Corrección de bandas

La corrección geométrica alinea todas las bandas espectrales y es **opcional**:

1. Abre la pestaña **Corrección** (se muestra la imagen principal).
2. Activa la selección de puntos y marca con **clic derecho** el mismo punto de referencia en
   cada banda (navegando con las flechas).
3. Pulsa **«Corregir bandas»** para alinearlas.

### Guardar y Entrenamiento

- **Guardar**: exporta los índices y la máscara de segmentación a un archivo `.mat`.
- **Entrenamiento**: permite configurar los hiperparámetros, actualizar la base de datos de
  imágenes y reentrenar los modelos AlexNet y AlgarroboNet.

## Funcionalidades

1. **Login**: autenticación con registro de usuarios.
2. **Visualización**: imágenes multiespectrales (R, G, B, NIR, RE y compuesta RGB), mostrando
   solo las bandas disponibles.
3. **Índices de vegetación**: vista de índice único con mapa de color, leyenda y valor por
   píxel (NDVI, GNDVI, NDRE, etc.).
4. **Corrección geométrica**: alineación de bandas espectrales.
5. **Segmentación manual**: trazado libre de ROI.
6. **Segmentación SAM**: segmentación multipunto con selección de máscara en el visor.
7. **Clasificación**: AlexNet y AlgarroboNet con tarjeta de resultado y probabilidades.
8. **Entrenamiento**: configuración y reentrenamiento de los modelos.
9. **Mapa**: ubicación georreferenciada con tiles satelitales.


## Requisitos Previos

- Python 3.10+
- Modelos descargados en `models/`

## Recursos Computacionales

La interfaz puede ejecutarse en **CPU** (todo funciona, pero la segmentación SAM y el
entrenamiento son lentos) o acelerarse con una **GPU NVIDIA con CUDA**, que es lo
recomendado para SAM y para reentrenar los modelos.

### Versiones del entorno

El entorno se ha probado y validado con las siguientes versiones:

| Componente | Versión |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.4.1 (build `+cu118`) |
| torchvision | 0.19.1 (build `+cu118`) |
| CUDA Toolkit (runtime de PyTorch) | 11.8 |
| cuDNN | 9.1 |
| ONNX Runtime (GPU) | 1.16.3 |

> La detección de GPU es automática: `config/settings.py` selecciona `cuda` si
> `torch.cuda.is_available()`, y recae en `cpu` en caso contrario. No es necesario
> configurar nada manualmente.

### Requisitos mínimos (modo CPU)

| Recurso | Mínimo |
|---|---|
| CPU | 4 núcleos x86-64 |
| RAM | 8 GB |
| Disco | ~3 GB libres (dependencias + modelos) |
| GPU | No requerida (SAM y entrenamiento funcionan en CPU, pero lentos) |

### Requisitos recomendados (con GPU)

| Recurso | Recomendado |
|---|---|
| CPU | 8 núcleos x86-64 |
| RAM | 16 GB |
| Disco | ~5 GB libres (SSD) |
| GPU | NVIDIA con **≥ 4 GB de VRAM** y soporte CUDA 11.8 |
| Driver NVIDIA | Compatible con CUDA 11.8 (≥ 520) |

> Entorno de referencia de desarrollo: NVIDIA **Quadro RTX 6000** (24 GB VRAM),
> driver con CUDA 11.8. Una tarjeta de ese nivel ejecuta SAM y el entrenamiento con
> holgura; 4–6 GB de VRAM son suficientes para la inferencia de SAM ViT-B y la
> clasificación con AlexNet/AlgarroboNet.

### Instalación de PyTorch con CUDA

PyTorch y torchvision **no** están en `requirements.txt` (dependen del hardware). Para
instalar la versión con CUDA 11.8:

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu==1.16.3
```

Para una instalación **solo CPU** (sin GPU NVIDIA):

```bash
pip install torch==2.4.1 torchvision==0.19.1
pip install onnxruntime==1.16.3
```

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

