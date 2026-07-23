import os
from pathlib import Path
from shutil import copy2

BASE_OLD = Path(r"D:\SOLEDAD VERSION FINAL\Z_DATA ANTINGUOS\Z_DATA ANTINGUOS")
BASE_NEW = Path(r"D:\SOLEDAD VERSION FINAL")

SPLITS = ["train", "test", "valid"]
VISIONS = ["NORMAL", "INFRARROJA", "NOCTURNO"]

ANIMAL_CONFIG = {
    "CONEJOS": [
        {
            "src": BASE_OLD / "Conejos.v5i.yolov11",
            "remap": {0: 0, 1: 0},
            "skip": set(),
        },
        {
            "src": BASE_OLD / "project vision.CONEJOSv2i.yolov11",
            "remap": {0: 0},
            "skip": set(),
        },
    ],
    "CUYES": [
        {
            "src": BASE_OLD / "Cuyes Silvestres.v1i.yolov11",
            "remap": {0: 0},
            "skip": set(),
        },
        {
            "src": BASE_OLD / "Segmentacion cuyes.v2i.yolov11",
            "remap": {0: 0},
            "skip": set(),
        },
    ],
    "GALLINAS": [
        {
            "src": BASE_OLD / "deteccion de gallinas.v2i.yolov11",
            "remap": {i: 0 for i in range(14)},
            "skip": set(),
        },
        {
            "src": BASE_OLD / "Gallinas.v1i.yolov11",
            "remap": {0: 0},
            "skip": set(),
        },
    ],
    "OVEJAS": [
        {
            "src": BASE_OLD / "OVEJAS_from_above.v3i.yolov11" / "sheeps_from_above.v3i.yolov11",
            "remap": {0: 0},
            "skip": set(),
        },
        {
            "src": BASE_OLD / "ovejas-y-guanacos-v2-migrated.v1i.yolov11",
            "remap": {1: 0},
            "skip": {0},
        },
    ],
    "ZORROS": [
        {
            "src": BASE_OLD / "dataset_zorros_limpio" / "dataset_zorros_limpio",
            "remap": {0: 0},
            "skip": set(),
        },
    ],
}

def remap_label_line(line: str, remap: dict, skip: set) -> str | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if not parts:
        return None
    try:
        old_class = int(parts[0])
    except ValueError:
        return None
    if old_class in skip:
        return None
    new_class = remap.get(old_class)
    if new_class is None:
        return None
    parts[0] = str(new_class)
    return " ".join(parts)

def process_animal(animal: str, configs: list):
    for split in SPLITS:
        label_count = 0
        for cfg in configs:
            src_labels = cfg["src"] / split / "labels"
            if not src_labels.exists():
                continue
            for label_file in src_labels.iterdir():
                if label_file.suffix.lower() != ".txt":
                    continue
                with open(label_file, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    result = remap_label_line(line, cfg["remap"], cfg["skip"])
                    if result is not None:
                        new_lines.append(result)
                content = "\n".join(new_lines)
                if content:
                    content += "\n"
                for vision in VISIONS:
                    dst_dir = BASE_NEW / animal / vision / split / "labels"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_file = dst_dir / label_file.name
                    with open(dst_file, "w") as f:
                        f.write(content)
                label_count += 1
        print(f"  {animal}/{split}: {label_count} labels copiados y remapeados")

def main():
    for animal, configs in ANIMAL_CONFIG.items():
        print(f"\n=== {animal} ===")
        process_animal(animal, configs)

if __name__ == "__main__":
    main()
