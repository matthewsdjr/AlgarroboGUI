FROM python:3.11-slim-bookworm

# Requisitos del sistema para GUI (Tkinter) y librerías nativas.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
#
# Si en tu PC tienes torch con CUDA (en tu caso cu118), agregamos el extra-index
# de PyTorch para que el build de la imagen pueda instalar las ruedas correctas.
#
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

COPY . .

ENV DISPLAY=:0

CMD ["python", "main.py"]
