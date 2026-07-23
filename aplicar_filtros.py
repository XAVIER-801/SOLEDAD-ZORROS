import cv2
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(r"D:\SOLEDAD VERSION FINAL")
ANIMALS = ["ZORROS", "OVEJAS", "GALLINAS", "CUYES", "CONEJOS"]
SPLITS = ["train", "test", "valid"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def infrared_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return colored

def night_vision_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.addWeighted(v, 1.0, np.zeros_like(v), 0, 0)
    v = np.clip(v * 0.7, 0, 255).astype(np.uint8)
    hsv_night = cv2.merge([h, s, v])
    img_dark = cv2.cvtColor(hsv_night, cv2.COLOR_HSV2BGR)
    night = img_dark.copy()
    night[:, :, 1] = np.clip(night[:, :, 1] * 1.4, 0, 255)
    night[:, :, 2] = np.clip(night[:, :, 2] * 0.6, 0, 255)
    noise = np.random.normal(0, 8, night.shape).astype(np.uint8)
    night = cv2.addWeighted(night, 1.0, noise, 0.15, 0)
    kernel = np.ones((2, 2), np.float32) / 4
    night = cv2.filter2D(night, -1, kernel)
    return night

def process_image(src_path, dst_infrared, dst_night):
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return f"FAIL read: {src_path.name}"
        h, w = img.shape[:2]
        if h > 0 and w > 0:
            ir = infrared_filter(img)
            cv2.imwrite(str(dst_infrared), ir)
            nv = night_vision_filter(img)
            cv2.imwrite(str(dst_night), nv)
            return f"OK: {src_path.name}"
        else:
            return f"SKIP invalid dims: {src_path.name}"
    except Exception as e:
        return f"ERR {src_path.name}: {e}"

def main():
    total = 0
    for animal in ANIMALS:
        src_normal = BASE / animal / "Normal"
        for split in SPLITS:
            src_dir = src_normal / split
            if not src_dir.exists():
                continue
            images = [p for p in src_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
            total += len(images)
            dst_ir_dir = BASE / animal / "INFRARROJA" / "Normal" / split
            dst_nt_dir = BASE / animal / "NOCTURNO" / "Normal" / split
            dst_ir_dir.mkdir(parents=True, exist_ok=True)
            dst_nt_dir.mkdir(parents=True, exist_ok=True)
            tasks = []
            for img_path in images:
                dst_ir = dst_ir_dir / img_path.name
                dst_nt = dst_nt_dir / img_path.name
                tasks.append((img_path, dst_ir, dst_nt))
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(process_image, *t): t[0] for t in tasks}
                for f in as_completed(futures):
                    pass
            ok = len([p for p in dst_ir_dir.iterdir() if p.suffix.lower() in EXTENSIONS])
            print(f"  {animal}/{split}: {ok} imagenes procesadas")
    print(f"\nTotal imagenes procesadas: {total}")

if __name__ == "__main__":
    main()
