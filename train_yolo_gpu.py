from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"GPU disponible: {torch.cuda.is_available()}")
    print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model = YOLO('yolo11n.pt')

    model.train(
        data=r'd:\SOLEDAD\dataset_yolo_consolidado\data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=r'd:\SOLEDAD\runs',
        name='detector_zorros_gpu',
        exist_ok=True,
        patience=20,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True
    )
    print('Entrenamiento completado.')
