"""
================================================================================
PIPELINE COMPLETO Y ESTANDARIZADO PARA DETECCIÓN DE ZORROS
================================================================================
Autores  : Pipeline generado para proyecto SOLEDAD
Fecha    : 2026-06-03
Objetivo : Construir un dataset limpio, consolidado y normalizado a partir de
           3 fuentes de datos heterogéneas, entrenar YOLO11 y evaluar con
           métricas completas para maximizar el rendimiento del modelo.

Estructura del pipeline:
    PASO 1  - Auditoría y diagnóstico de cada dataset original
    PASO 2  - Limpieza y estandarización (seg → bbox, folder → YOLO)
    PASO 3  - Consolidación en un único dataset limpio
    PASO 4  - Generación del data.yaml correcto
    PASO 5  - Entrenamiento con hiperparámetros optimizados
    PASO 6  - Validación y métricas completas
    PASO 7  - Visualización de resultados
================================================================================
"""

import os
import sys
import glob
import shutil
import random
import json
import matplotlib
matplotlib.use('Agg')  # Para entornos sin pantalla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN GLOBAL — MODIFICA SOLO AQUÍ
# ============================================================
WORKSPACE = r"d:\SOLEDAD"
OUTPUT_DATASET = os.path.join(WORKSPACE, "dataset_zorros_limpio")
RUNS_DIR       = os.path.join(WORKSPACE, "runs")
RUN_NAME       = "detector_zorros_final"
IMGSZ          = 640
EPOCHS         = 100
BATCH          = 16
DEVICE         = 0          # 0 = primera GPU; 'cpu' si no tienes GPU
PATIENCE       = 20         # Early stopping

# Rutas de los 3 datasets originales
DS1_PATH = os.path.join(WORKSPACE, "DATOS ANTIGUOS DESCARGADO", "Sorros-yolo.v3i.yolov11")
DS2_PATH = os.path.join(WORKSPACE, "DATOS ANTIGUOS DESCARGADO", "Zorros.v2-zorro.yolov11")
DS3_PATH = os.path.join(WORKSPACE, "DATOS ANTIGUOS DESCARGADO", "fox.v2i.folder")

# ============================================================
# UTILIDADES
# ============================================================

def log(msg, level="INFO"):
    prefix = {"INFO": "✔", "WARN": "⚠", "ERROR": "✘", "STEP": "▶"}
    print(f"  [{prefix.get(level,'·')}] {msg}")

def banner(title):
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")

# ============================================================
# PASO 1: AUDITORÍA DE DATASETS ORIGINALES
# ============================================================

def auditar_datasets():
    banner("PASO 1: AUDITORÍA DE DATASETS ORIGINALES")
    resumen = {}

    for nombre, ruta in [("Sorros-yolo (DS1)", DS1_PATH),
                          ("Zorros.v2 (DS2)",   DS2_PATH),
                          ("fox.v2i (DS3)",      DS3_PATH)]:
        log(f"Analizando: {nombre}", "STEP")
        info = {"ruta": ruta, "splits": {}}

        for split in ["train", "valid", "test"]:
            # DS3 tiene estructura diferente: train/fox/
            if nombre.startswith("fox"):
                img_dir = os.path.join(ruta, split, "fox")
                lbl_dir = None  # Las etiquetas están embebidas como XML/JSON en folder format
            else:
                img_dir = os.path.join(ruta, split, "images")
                lbl_dir = os.path.join(ruta, split, "labels")

            n_imgs = len(glob.glob(os.path.join(img_dir, "*.*"))) if os.path.exists(img_dir) else 0
            n_lbls = len(glob.glob(os.path.join(lbl_dir, "*.txt"))) if (lbl_dir and os.path.exists(lbl_dir)) else 0

            if n_imgs > 0:
                # Detectar tipo de anotación por tamaño promedio de archivos label
                tipo = "N/A"
                if lbl_dir and os.path.exists(lbl_dir):
                    txts = glob.glob(os.path.join(lbl_dir, "*.txt"))[:10]
                    if txts:
                        avg_size = np.mean([os.path.getsize(t) for t in txts])
                        tipo = "Segmentación (polígono)" if avg_size > 200 else "BBox YOLO"

                info["splits"][split] = {"imágenes": n_imgs, "etiquetas": n_lbls, "tipo": tipo}
                log(f"  {split:6s}: {n_imgs:4d} imágenes | {n_lbls:4d} etiquetas | {tipo}")

        resumen[nombre] = info

    print()
    log("OBSERVACIÓN CRÍTICA:", "WARN")
    log("DS1 (Sorros-yolo): Anotaciones en SEGMENTACIÓN → se convertirán a BBox", "WARN")
    log("DS2 (Zorros.v2):   Anotaciones en BBOX YOLO   → uso directo ✔", "INFO")
    log("DS3 (fox.v2i):     Formato 'folder' sin labels → se excluirá automáticamente", "WARN")
    return resumen

# ============================================================
# PASO 2: CONVERSIÓN DE SEGMENTACIÓN → BBOX
# ============================================================

def seg_a_bbox(coords):
    """Convierte lista de puntos de polígono [x1,y1,x2,y2,...] a bbox centrado."""
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs or not ys:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_c = (xmin + xmax) / 2.0
    y_c = (ymin + ymax) / 2.0
    w   = xmax - xmin
    h   = ymax - ymin
    # Validar que el bbox sea razonable
    if w <= 0 or h <= 0 or w > 1 or h > 1:
        return None
    return x_c, y_c, w, h

def procesar_etiqueta(src, dst, convertir_seg=False):
    """
    Lee un archivo de etiquetas YOLO y lo escribe normalizado.
    Si convertir_seg=True, convierte polígonos a bbox.
    """
    try:
        with open(src, 'r') as f:
            lines = f.readlines()
    except Exception:
        return False

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = 0  # Todo es clase 'zorro'
            coords = [float(x) for x in parts[1:]]

            if len(coords) == 4:
                # Ya es un bbox YOLO normalizado
                xc, yc, w, h = coords
                if 0 < w <= 1 and 0 < h <= 1:
                    out_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            elif len(coords) > 4 and convertir_seg:
                # Polígono → convertir a bbox
                result = seg_a_bbox(coords)
                if result:
                    xc, yc, w, h = result
                    out_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        except Exception:
            continue

    if out_lines:
        with open(dst, 'w') as f:
            f.writelines(out_lines)
        return True
    return False

# ============================================================
# PASO 3: CONSOLIDACIÓN LIMPIA DE DATASETS
# ============================================================

def consolidar_datasets(output_dir):
    banner("PASO 2-3: CONSOLIDACIÓN Y NORMALIZACIÓN")

    # Limpiar directorio previo
    if os.path.exists(output_dir):
        log(f"Eliminando dataset previo en: {output_dir}", "WARN")
        shutil.rmtree(output_dir)

    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    stats = {s: {"ok": 0, "skip": 0} for s in splits}

    # ---- DS1: Sorros-yolo (segmentación → bbox) ----
    log("Procesando DS1 (Sorros-yolo) — convirtiendo polígonos a bbox...", "STEP")
    for split in ["train", "valid", "test"]:
        src_imgs = os.path.join(DS1_PATH, split, "images")
        src_lbls = os.path.join(DS1_PATH, split, "labels")
        dst_split = "valid" if split == "valid" else split
        dst_imgs  = os.path.join(output_dir, dst_split, "images")
        dst_lbls  = os.path.join(output_dir, dst_split, "labels")

        if not os.path.exists(src_imgs):
            continue

        for img_path in glob.glob(os.path.join(src_imgs, "*.*")):
            name = Path(img_path).stem
            lbl_path = os.path.join(src_lbls, name + ".txt")

            if not os.path.exists(lbl_path):
                stats[dst_split]["skip"] += 1
                continue

            new_img = os.path.join(dst_imgs, f"ds1_{Path(img_path).name}")
            new_lbl = os.path.join(dst_lbls, f"ds1_{name}.txt")
            shutil.copy2(img_path, new_img)
            ok = procesar_etiqueta(lbl_path, new_lbl, convertir_seg=True)
            if ok:
                stats[dst_split]["ok"] += 1
            else:
                os.remove(new_img)
                stats[dst_split]["skip"] += 1

    # ---- DS2: Zorros.v2-zorro (bbox directo, mejor calidad) ----
    log("Procesando DS2 (Zorros.v2) — formato bbox directo...", "STEP")
    for split in ["train", "valid", "test"]:
        src_imgs = os.path.join(DS2_PATH, split, "images")
        src_lbls = os.path.join(DS2_PATH, split, "labels")
        dst_split = "valid" if split == "valid" else split
        dst_imgs  = os.path.join(output_dir, dst_split, "images")
        dst_lbls  = os.path.join(output_dir, dst_split, "labels")

        if not os.path.exists(src_imgs):
            continue

        for img_path in glob.glob(os.path.join(src_imgs, "*.*")):
            name = Path(img_path).stem
            lbl_path = os.path.join(src_lbls, name + ".txt")

            if not os.path.exists(lbl_path):
                stats[dst_split]["skip"] += 1
                continue

            new_img = os.path.join(dst_imgs, f"ds2_{Path(img_path).name}")
            new_lbl = os.path.join(dst_lbls, f"ds2_{name}.txt")
            shutil.copy2(img_path, new_img)
            ok = procesar_etiqueta(lbl_path, new_lbl, convertir_seg=False)
            if ok:
                stats[dst_split]["ok"] += 1
            else:
                os.remove(new_img)
                stats[dst_split]["skip"] += 1

    # ---- DS3: fox.v2i.folder — NO tiene labels YOLO → se excluye ----
    log("DS3 (fox.v2i.folder): excluido (no contiene labels en formato YOLO)", "WARN")

    # ---- Resumen ----
    total = sum(v["ok"] for v in stats.values())
    log(f"\n  Consolidación completada: {total} pares imagen-etiqueta válidos", "INFO")
    for s, v in stats.items():
        log(f"    {s:6s}: {v['ok']:4d} válidos | {v['skip']:3d} omitidos")

    return stats

# ============================================================
# PASO 4: GENERAR data.yaml CORRECTO
# ============================================================

def crear_yaml(output_dir):
    banner("PASO 4: GENERANDO data.yaml")
    yaml_path = os.path.join(output_dir, "data.yaml")

    # Contar imágenes por split para reporte
    for split in ["train", "valid", "test"]:
        n = len(glob.glob(os.path.join(output_dir, split, "images", "*.*")))
        log(f"  {split:6s}: {n} imágenes")

    contenido = f"""# Dataset de zorros consolidado y limpio
# Generado automáticamente por pipeline_zorros_completo.py
train: {os.path.join(output_dir, 'train', 'images')}
val:   {os.path.join(output_dir, 'valid', 'images')}
test:  {os.path.join(output_dir, 'test',  'images')}

nc: 1
names:
  0: zorro
"""
    with open(yaml_path, 'w') as f:
        f.write(contenido)

    log(f"data.yaml creado en: {yaml_path}", "INFO")
    return yaml_path

# ============================================================
# PASO 5: ENTRENAMIENTO CON HIPERPARÁMETROS OPTIMIZADOS
# ============================================================

def entrenar(yaml_path):
    banner("PASO 5: ENTRENAMIENTO YOLO11")
    log(f"  Modelo   : yolo11n.pt (YOLO11 Nano preentrenado en COCO)")
    log(f"  Épocas   : {EPOCHS}")
    log(f"  Imgsz    : {IMGSZ}")
    log(f"  Batch    : {BATCH}")
    log(f"  Device   : {DEVICE}")
    log(f"  Patience : {PATIENCE} (early stopping)")
    log(f"  Run name : {RUN_NAME}")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data       = yaml_path,
        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = BATCH,
        device     = DEVICE,
        project    = RUNS_DIR,
        name       = RUN_NAME,
        exist_ok   = True,
        patience   = PATIENCE,
        # Augmentación fuerte para mayor generalización
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.5,
        flipud     = 0.5,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,
        # Regularización
        weight_decay = 0.0005,
        warmup_epochs = 3.0,
        # Optimizador
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        # Guardar resultados
        save       = True,
        save_period = 10,
        val        = True,
        plots      = True,
        verbose    = True,
    )

    log("Entrenamiento completado.", "INFO")
    return model, results

# ============================================================
# PASO 6: VALIDACIÓN COMPLETA CON MÉTRICAS
# ============================================================

def validar_y_reportar(model, run_dir):
    banner("PASO 6: VALIDACIÓN COMPLETA Y MÉTRICAS")

    log("Ejecutando validación sobre el conjunto de test...", "STEP")
    metrics = model.val(split="test", plots=True)

    # Métricas principales
    mAP50    = metrics.box.map50
    mAP5095  = metrics.box.map
    precision = metrics.box.mp
    recall    = metrics.box.mr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print()
    print("  " + "═" * 60)
    print("  ║  RESULTADOS DE VALIDACIÓN — DETECTOR DE ZORROS            ║")
    print("  " + "═" * 60)
    print(f"  ║  mAP@50        : {mAP50:.4f}  ({mAP50*100:.2f}%)                      ║")
    print(f"  ║  mAP@50-95     : {mAP5095:.4f}  ({mAP5095*100:.2f}%)                      ║")
    print(f"  ║  Precisión     : {precision:.4f}  ({precision*100:.2f}%)                      ║")
    print(f"  ║  Recall        : {recall:.4f}  ({recall*100:.2f}%)                      ║")
    print(f"  ║  F1-Score      : {f1:.4f}  ({f1*100:.2f}%)                      ║")
    print("  " + "═" * 60)

    # Métricas por clase
    print()
    log("Métricas por clase:", "STEP")
    names = model.names
    clases_eval = getattr(metrics, 'classes', list(range(len(metrics.box.p))))
    for i, c in enumerate(clases_eval):
        nombre_clase = names.get(int(c), f"Clase {c}")
        p  = metrics.box.p[i]  if i < len(metrics.box.p)  else 0.0
        r  = metrics.box.r[i]  if i < len(metrics.box.r)  else 0.0
        ap50   = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
        ap5095 = metrics.box.ap[i]   if i < len(metrics.box.ap)   else 0.0
        fi = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
        log(f"  Clase '{nombre_clase}': P={p:.4f} | R={r:.4f} | F1={fi:.4f} | AP50={ap50:.4f} | AP50-95={ap5095:.4f}")

    # Guardar métricas en JSON
    reporte = {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "run_dir": run_dir,
    }
    reporte_path = os.path.join(run_dir, "reporte_metricas.json")
    with open(reporte_path, "w") as f:
        json.dump(reporte, f, indent=4)
    log(f"Reporte JSON guardado en: {reporte_path}", "INFO")

    return metrics, reporte

# ============================================================
# PASO 7: VISUALIZACIÓN DE RESULTADOS
# ============================================================

def visualizar_resultados(run_dir, reporte):
    banner("PASO 7: VISUALIZACIÓN DE RESULTADOS")

    plots_info = [
        ("Diagrama de Aprendizaje (Pérdidas y mAP)", "results.png"),
        ("Matriz de Confusión",                       "confusion_matrix.png"),
        ("Matriz de Confusión Normalizada",            "confusion_matrix_normalized.png"),
        ("Curva Precisión-Recall (PR)",               "PR_curve.png"),
        ("Curva F1-Confianza",                        "F1_curve.png"),
        ("Curva Precisión-Confianza",                 "P_curve.png"),
        ("Curva Recall-Confianza",                    "R_curve.png"),
    ]

    for titulo, filename in plots_info:
        ruta = os.path.join(run_dir, filename)
        if os.path.exists(ruta):
            log(f"Mostrando: {titulo}", "INFO")
            try:
                img = Image.open(ruta)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(titulo, fontsize=15, fontweight='bold', pad=12)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                log(f"No se pudo mostrar {filename}: {e}", "WARN")
        else:
            log(f"No encontrado: {filename}", "WARN")

    # ---- Gráfico resumen de métricas ----
    metricas_nombres = ["mAP@50", "mAP@50-95", "Precisión", "Recall", "F1-Score"]
    metricas_vals    = [reporte["mAP50"], reporte["mAP50_95"],
                        reporte["precision"], reporte["recall"], reporte["f1_score"]]
    colores = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.5 else "#F44336" for v in metricas_vals]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(metricas_nombres, metricas_vals, color=colores)
    plt.xlim(0, 1)
    for bar, val in zip(bars, metricas_vals):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f} ({val*100:.1f}%)",
                 va='center', fontsize=11, fontweight='bold')
    plt.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Objetivo 70%')
    plt.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Aceptable 50%')
    leyenda = [mpatches.Patch(color='#4CAF50', label='≥ 70% (Bueno)'),
               mpatches.Patch(color='#FF9800', label='50-70% (Aceptable)'),
               mpatches.Patch(color='#F44336', label='< 50% (Bajo)')]
    plt.legend(handles=leyenda, loc='lower right')
    plt.title("Resumen de Métricas — Detector de Zorros", fontsize=14, fontweight='bold')
    plt.xlabel("Valor (0 - 1)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "resumen_metricas.png"), dpi=150)
    plt.show()
    log(f"Gráfico resumen guardado en: {os.path.join(run_dir, 'resumen_metricas.png')}", "INFO")

# ============================================================
# MAIN — EJECUCIÓN COMPLETA
# ============================================================

def main():
    banner("PIPELINE DETECCIÓN DE ZORROS — INICIO")
    log(f"Workspace   : {WORKSPACE}")
    log(f"Dataset dest: {OUTPUT_DATASET}")
    log(f"Runs dir    : {RUNS_DIR}")

    # --- PASO 1: Auditoría ---
    auditar_datasets()

    # --- PASO 2-3: Consolidación ---
    consolidar_datasets(OUTPUT_DATASET)

    # --- PASO 4: data.yaml ---
    yaml_path = crear_yaml(OUTPUT_DATASET)

    # --- PASO 5: Entrenamiento ---
    model, train_results = entrenar(yaml_path)

    # Ruta del run
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    # --- PASO 6: Validación ---
    metrics, reporte = validar_y_reportar(model, run_dir)

    # --- PASO 7: Visualización ---
    visualizar_resultados(run_dir, reporte)

    banner("PIPELINE COMPLETADO EXITOSAMENTE")
    log(f"Modelo final: {os.path.join(run_dir, 'weights', 'best.pt')}", "INFO")
    log(f"Reporte JSON: {os.path.join(run_dir, 'reporte_metricas.json')}", "INFO")
    log(f"mAP@50  = {reporte['mAP50']*100:.2f}%", "INFO")
    log(f"F1-Score = {reporte['f1_score']*100:.2f}%", "INFO")


if __name__ == "__main__":
    main()
