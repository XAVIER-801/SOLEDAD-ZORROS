# Usar una imagen base de Python 3.12 slim
FROM python:3.12-slim

# Evitar que Python escriba archivos .pyc en disco y habilitar el buffer inmediato de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y utilidades básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias del proyecto
COPY requirements.txt .

# Instalar torch CPU-only primero (evita descargar CUDA de ~900MB innecesario)
RUN pip install --no-cache-dir --timeout=120 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias del proyecto
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# Copiar el resto del código del proyecto al contenedor
COPY . .

# Exponer el puerto en el que corre la aplicación FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI usando Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
