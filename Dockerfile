FROM python:3.10-slim

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and necessary datasets
COPY app/ ./app/
COPY dataset_yolo_consolidado/ ./dataset_yolo_consolidado/
# If weights exist, copy them, otherwise YOLO will download the default yolo11n.pt/yolov8n.pt at runtime
COPY runs/ ./runs/

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
