# 🦊 GUARDIÁN CONTRA ZORROS — Sistema de Vigilancia Inteligente para Protección de Animales de Granja

## 1. INTRODUCCIÓN Y NECESIDAD DEL PROYECTO

La **depredación por zorros** representa una amenaza constante para diversas especies animales en entornos rurales y de granja. Los zorros son depredadores oportunistas que atacan:

- **Ovinos** (corderos recién nacidos, especialmente durante la temporada de parición)
- **Aves de corral** (gallinas, pollos, patos en gallineros)
- **Cuyes / conejos** (en cuyerías y conejeras)
- **Otros animales pequeños** en criaderos abiertos o semiabiertos

Los métodos tradicionales de protección (pastores nocturnos, cercas, perros guardianes) son costosos, limitados o no están disponibles para pequeños y medianos productores.

**Guardían Contra Zorros** propone una solución de **visión por computadora en tiempo real** utilizando un modelo YOLO11 (You Only Look Once) entrenado específicamente para detectar zorros mediante una cámara web, activando alertas sonoras y visuales para ahuyentar al depredador y notificar al cuidador.

### 1.1 Problemática

- Ataques nocturnos de zorros a corderos, gallinas, cuyes y otras presas
- Pérdidas económicas significativas para pequeños y medianos productores
- Ausencia de sistemas automatizados de bajo costo y fácil instalación
- Dificultad de monitoreo humano 24/7 en múltiples puntos vulnerables
- Estrés y muerte de animales en gallineros, cuyerías y criaderos abiertos

### 1.2 Solución Propuesta

Sistema de detección automatizada que:
1. Captura video en tiempo real mediante cámara web o cámara IP
2. Procesa cada frame con un detector YOLO11 entrenado para reconocer zorros
3. Dibuja cuadros delimitadores (bounding boxes) sobre los zorros detectados
4. Activa sirenas y alertas de voz para ahuyentar al depredador
5. Notifica al cuidador vía interfaz web accesible desde cualquier dispositivo
6. Funciona en **gallineros, cuyerías, corrales de ovejas o cualquier criadero abierto**

---

## 2. OBJETIVOS

### Objetivo General
Desarrollar e implementar un sistema de detección automática de zorros basado en visión por computadora (YOLO11) para la protección de animales de granja (ovinos, aves de corral, cuyes, conejos y otros) vulnerables a ataques de zorros.

### Objetivos Específicos

1. **Recopilar y consolidar** datasets heterogéneos de zorros provenientes de Roboflow
2. **Estandarizar y limpiar** las anotaciones (convertir polígonos a bounding boxes, eliminar datos sin etiquetas)
3. **Entrenar un modelo YOLO11 Nano** con alto rendimiento (mAP@50 ≥ 90%)
4. **Implementar una interfaz web** con video en vivo, detección y alertas
5. **Desplegar el sistema** mediante Docker para fácil instalación en cualquier computadora con GPU

---

## 3. METODOLOGÍA — PIPELINE COMPLETO

### 3.1 Descarga de Datasets (Fuentes)

Se descargaron **3 datasets** desde la plataforma **Roboflow**:

| Dataset | Formato | Imágenes | Anotaciones | Calidad |
|---------|---------|----------|-------------|---------|
| **DS1:** `Sorros-yolo.v3i.yolov11` | YOLO con segmentación (polígonos) | 237 train, 12 valid, 12 test | Polígonos → requieren conversión a bbox | Media |
| **DS2:** `Zorros.v2-zorro.yolov11` | YOLO con bounding boxes | 400 train, 40 valid, 20 test | BBox directo, listo para usar | **Alta** |
| **DS3:** `fox.v2i.folder` | Formato folder de Roboflow | 531 train (sin labels YOLO) | Sin etiquetas en formato YOLO | **Excluido** |

> **DS3 fue excluido** del entrenamiento por no contener etiquetas en formato YOLO `.txt` compatibles.

### 3.2 Diagrama del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DS1 ──► AUDITORÍA ──► CONVERSIÓN (seg→bbox) ──┐               │
│  DS2 ──► AUDITORÍA ──► NORMALIZACIÓN ───────────┤               │
│  DS3 ──► AUDITORÍA ──► EXCLUIDO (sin labels) ───┘               │
│                                                     ▼           │
│                                          CONSOLIDACIÓN           │
│                                          (dataset_zorros_limpio)│
│                                                     ▼           │
│                                          GENERAR data.yaml      │
│                                                     ▼           │
│                                          ENTRENAMIENTO YOLO11   │
│                                          (yolo11n.pt, 100 épocas)│
│                                                     ▼           │
│                                          VALIDACIÓN + MÉTRICAS  │
│                                                     ▼           │
│                                          WEB APP (Flask + HTML) │
│                                          + Alertas (sirena/voz) │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Paso 1: Auditoría de Datasets

Se analizó cada dataset para determinar:
- **Tipo de anotación**: segmentación (polígono) vs. bounding box
- **Cantidad** de imágenes y etiquetas por split (train/valid/test)
- **Compatibilidad** con formato YOLO

**Hallazgos críticos:**
- **DS1:** Anotaciones en formato de segmentación (polígonos con múltiples puntos) → requieren conversión a bounding boxes
- **DS2:** Anotaciones en formato YOLO BBox directo → uso inmediato
- **DS3:** Formato "folder" de Roboflow sin archivos `.txt` de etiquetas → **excluido automáticamente**

### 3.4 Paso 2: Conversión Segmentación → Bounding Box

Para DS1, se implementó una función `seg_a_bbox()` que:
1. Extrae las coordenadas `[x1, y1, x2, y2, ..., xn, yn]` del polígono
2. Calcula el bounding box mínimos: `x_min, y_min, x_max, y_max`
3. Convierte a formato YOLO normalizado: `(x_centro, y_centro, ancho, alto)`
4. Valida que las dimensiones del bbox sean positivas y estén en rango `[0, 1]`

```
Ejemplo de conversión:
┌──────────────────┐         ┌──────────────────┐
│  Polígono:        │    →    │  Bounding Box:    │
│  x1,y1,x2,y2,... │         │  xc yc w h        │
│  0.5 0.3 0.6 0.4 │         │  0.55 0.35 0.2 0.2│
│  0.7 0.4 0.5 0.2 │         └──────────────────┘
└──────────────────┘
```

### 3.5 Paso 3: Consolidación en Dataset Limpio

Se creó `dataset_zorros_limpio/` con la siguiente estructura:

```
dataset_zorros_limpio/
├── train/
│   ├── images/    (637 imágenes — ds1_* + ds2_*)
│   └── labels/    (637 etiquetas .txt en formato YOLO)
├── valid/
│   ├── images/    (52 imágenes)
│   └── labels/    (52 etiquetas)
├── test/
│   ├── images/    (32 imágenes)
│   └── labels/    (32 etiquetas)
└── data.yaml      (configuración del dataset)
```

**Total: 721 pares imagen-etiqueta válidos** (solo DS1 + DS2, limpios y normalizados).

**Prefijos:** `ds1_` para imágenes de DS1, `ds2_` para imágenes de DS2 (evita colisiones de nombres).

### 3.6 Paso 4: Configuración del Dataset (`data.yaml`)

```yaml
train: D:\SOLEDAD\dataset_zorros_limpio\train\images
val:   D:\SOLEDAD\dataset_zorros_limpio\valid\images
test:  D:\SOLEDAD\dataset_zorros_limpio\test\images

nc: 1  # Número de clases: 1 (solo "zorro")
names:
  0: zorro
```

### 3.7 Paso 5: Entrenamiento del Modelo YOLO11

**Arquitectura:** YOLO11n (Nano) — la versión más ligera y rápida de YOLO11, ideal para inferencia en tiempo real en computadoras de gama media.

**Hiperparámetros optimizados:**

| Parámetro | Valor | Explicación |
|-----------|-------|-------------|
| Modelo base | `yolo11n.pt` | Preentrenado en COCO (transfer learning) |
| Épocas | 100 | Suficientes para convergencia |
| Tamaño imagen | 640×640 | Balance velocidad-precisión |
| Batch | 16 | Ajustado a GPU con 8GB VRAM |
| Optimizador | **AdamW** | Mejor convergencia que SGD |
| Learning rate | 0.001 | Tasa inicial estándar |
| Weight decay | 0.0005 | Regularización L2 |
| Early stopping | paciencia=20 | Detiene si no mejora en 20 épocas |
| Device | GPU 0 | RTX 3050 / GPU compatible CUDA |

**Aumentación de datos (data augmentation):**

| Técnica | Valor | Efecto |
|---------|-------|--------|
| HSV (Hue/Sat/Value) | 0.015 / 0.7 / 0.4 | Variación de color y brillo |
| Rotación | ±10° | Diferentes ángulos de cámara |
| Traslación | ±10% | Diferentes posiciones en frame |
| Escala | ±50% | Zorros a diferentes distancias |
| Volteo horizontal | 50% | Simular zorros en ambas direcciones |
| Volteo vertical | 50% | Simular distintas alturas de cámara |
| Mosaico | 100% | Combina 4 imágenes en 1 (mejora detección en contexto) |
| MixUp | 10% | Mezcla imágenes para mejor generalización |

### 3.8 Paso 6: Validación y Métricas

Se evaluó el modelo en el conjunto de **test** (32 imágenes nunca vistas durante el entrenamiento).

### 3.9 Paso 7: Aplicación Web (Flask)

Interfaz web desarrollada con **Flask** (Python) que permite:
- **Visualizar** video en vivo desde la cámara web
- **Detección** en tiempo real con bounding boxes
- **Alertas sonoras**: sirena activada al detectar un zorro
- **Alertas de voz**: mensaje "¡Se ha detectado un zorro!" en español
- **Estadísticas** de detección en pantalla

```
Arquitectura de la aplicación web:

  Cámara Web ──► Frame ──► YOLO11 Model ──► Bounding Boxes ──► Video Stream
                              │
                              ▼
                    ¿Zorro detectado?
                         /    \
                       SÍ     NO
                       /        \
                  Activar       └─► Mostrar solo video
                  Sirena
                  + Voz
                  + Alerta
```

**Despliegue con Docker:**

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "5000:5000"
    devices:
      - /dev/video0  # Cámara web
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # Aceleración GPU
```

---

## 4. RESULTADOS OBTENIDOS

### 4.1 Métricas del Modelo Final

| Métrica | Valor | Interpretación |
|---------|-------|---------------|
| **mAP@50** | **98.60%** | ✅ Excelente — el modelo detecta zorros con alta precisión |
| **mAP@50-95** | **72.83%** | ✅ Muy bueno — buen ajuste de bounding boxes |
| **Precisión (Precision)** | **96.40%** | ✅ Baja tasa de falsos positivos |
| **Recall (Exhaustividad)** | **97.06%** | ✅ Baja tasa de falsos negativos |
| **F1-Score** | **96.73%** | ✅ Balance perfecto entre precisión y recall |

### 4.2 Interpretación de Resultados

| Métrica | ¿Qué significa? | ¿Qué indica nuestro valor? |
|---------|----------------|---------------------------|
| **mAP@50 (98.60%)** | Porcentaje de detecciones correctas con IoU ≥ 50% | De cada 100 zorros en las imágenes, el modelo detecta correctamente ~99 |
| **Precisión (96.40%)** | De todas las detecciones, cuántas son realmente zorros | Solo ~4 de cada 100 detecciones son falsas alarmas |
| **Recall (97.06%)** | De todos los zorros presentes, cuántos fueron detectados | Solo ~3 de cada 100 zorros pasan desapercibidos |
| **F1-Score (96.73%)** | Media armónica de precisión y recall | Rendimiento global excelente |

**Conclusión:** El modelo es **altamente confiable** para su uso en producción. La combinación de alta precisión y alto recall significa que:
- **No habrá falsas alarmas frecuentes** que molesten al ganadero (precisión 96.4%)
- **Casi ningún zorro pasará desapercibido** (recall 97.06%)

### 4.3 Hardware Utilizado

| Componente | Especificación |
|------------|---------------|
| CPU | Intel i7 / AMD Ryzen (multinúcleo) |
| GPU | NVIDIA RTX 3050 (4GB VRAM) o superior |
| RAM | 16 GB |
| SO | Windows 10/11 |
| Cámara | Webcam USB 1080p |

**Capacidad de la computadora aprovechada:**
- **GPU (CUDA):** Entrenamiento 20-30x más rápido que CPU. El modelo YOLO11n está optimizado para inferencia en GPU a ~30-60 FPS
- **CPU multihilo:** Procesamiento de 8 workers para carga de datos durante entrenamiento
- **RAM:** Suficiente para mantener el dataset en memoria caché y acelerar épocas

---

## 5. TRABAJO FUTURO — MEJORAS PROPUESTAS

### 5.1 Entrenar con Imágenes de Ovejas

Para mejorar la **discriminación zorro vs. oveja** y reducir falsos positivos, se propone:

1. **Recolectar un dataset de ovejas** en el mismo entorno (misma cámara, misma hora del día)
2. **Incorporar ovejas como segunda clase** en el modelo (nc=2: "zorro", "oveja")
3. **Aumentar el dataset** con imágenes nocturnas e infrarrojas
4. **Re-entrenar** con las siguientes configuraciones:

```
Dataset ampliado propuesto:
├── zorro/     (721 imágenes existentes + 300 nuevas = ~1000)
├── oveja/     (500 imágenes nuevas con etiquetas)
├── fondo/     (200 imágenes sin animales — reduce falsos positivos)
└── data.yaml  (nc: 2 — zorro, oveja)
```

### 5.2 Otras Mejoras Potenciales

| Mejora | Beneficio | Complejidad |
|--------|-----------|-------------|
| **Agregar clase "oveja"** | Discriminar zorro de oveja | Media |
| **Detección nocturna** | Funcionar 24/7 con cámara infrarroja | Alta |
| **Notificaciones SMS/WhatsApp** | Alertar al pastor remotamente | Baja |
| **Seguimiento (tracking)** | Evitar alertas repetitivas del mismo zorro | Media |
| **Historial de detecciones** | Reportes semanales de actividad | Baja |
| **Detección de múltiples zorros** | Manejar ataques en manada | Media |
| **Transmisión por radio** | Activar cercos eléctricos | Alta |

### 5.3 Recomendaciones para Producción

1. **Cámara IP nocturna** con visión infrarroja (alcance 20-30m)
2. **Altavoz direccional** para alertas sonoras sin molestar al ganado
3. **Batería de respaldo** + panel solar para operación autónoma
4. **Actualización del modelo** vía OTA (over-the-air) cuando se mejore
5. **Almacenamiento local** de detecciones en SQLite para análisis posterior

---

## 6. ESTRUCTURA DEL REPOSITORIO

```
D:\SOLEDAD\
├── app/
│   ├── main.py                 # Aplicación Flask (servidor web + detección)
│   ├── templates/
│   │   └── index.html          # Interfaz de usuario (video + alertas)
│   └── static/                 # CSS, JS, imágenes (si aplica)
├── dataset_zorros_limpio/      # Dataset limpio y normalizado (721 imágenes)
│   ├── train/images+labels/    # 637 muestras de entrenamiento
│   ├── valid/images+labels/    # 52 muestras de validación
│   ├── test/images+labels/     # 32 muestras de prueba
│   └── data.yaml               # Configuración del dataset
├── dataset_yolo_consolidado/   # Dataset completo (incluye DS3 sin etiquetas)
├── DATOS ANTIGUOS DESCARGADO/  # Datasets originales de Roboflow
│   ├── Sorros-yolo.v3i.yolov11/       # DS1 (polígonos)
│   ├── Zorros.v2-zorro.yolov11/       # DS2 (bbox)
│   └── fox.v2i.folder/                # DS3 (excluido)
├── runs/
│   ├── detector_zorros/         # Primer entrenamiento (sin optimizar)
│   ├── detector_zorros-5/       # Entrenamiento intermedio
│   └── detector_zorros_final/   # ✅ Modelo final entrenado
│       ├── weights/
│       │   ├── best.pt          # ✅ Mejor modelo (mAP@50 = 98.60%)
│       │   └── last.pt          # Último checkpoint
│       ├── reporte_metricas.json # Métricas en formato JSON
│       ├── results.png          # Curvas de aprendizaje
│       ├── confusion_matrix.png # Matriz de confusión
│       └── *.jpg                # Predicciones de validación
├── pipeline_zorros_completo.py  # Pipeline completo (7 pasos)
├── train_yolo_gpu.py           # Script de entrenamiento alternativo
├── entrenamiento_yolo.ipynb    # Notebook Jupyter (versión 1)
├── entrenamiento_yolo_v2.ipynb # Notebook Jupyter (versión 2)
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Construcción de imagen Docker
├── docker-compose.yml          # Orquestación Docker + GPU
├── run_tunnel.bat              # Script para túnel ngrok
├── yolo11n.pt                  # Modelo base preentrenado
└── README.md                   # Este archivo
```

---

## 7. CONCLUSIONES

El sistema **Guardían Contra Zorros** demuestra que es posible construir un detector de zorros de **alto rendimiento** (mAP@50 = 98.60%) con recursos accesibles:

- **Solo 721 imágenes** de zorros fueron suficientes gracias a transfer learning (YOLO11 preentrenado en COCO) y una fuerte aumentación de datos
- El **pipeline automatizado** permite reproducir todo el flujo desde cero: auditoría → limpieza → consolidación → entrenamiento → validación
- La **aplicación web** despliega el modelo en tiempo real con alertas visuales y sonoras
- **Docker** facilita el despliegue en cualquier computadora con GPU NVIDIA
- Es aplicable en **múltiples escenarios**: corrales de ovejas, gallineros, cuyerías, conejeras y cualquier criadero abierto vulnerable a zorros

El sistema está listo para pruebas de campo y puede ser mejorado incrementalmente agregando más datos (especialmente imágenes de ovejas, gallinas y cuyes para discriminación multiclase) y adaptándolo a condiciones nocturnas.

---

## 8. CRÉDITOS

- **Desarrollado por:** XAVIER-801
- **Datasets:** Roboflow Universe (licencias originales)
- **Framework:** Ultralytics YOLO11
- **Web framework:** Flask (Python)
- **Contenerización:** Docker + docker-compose

---

*"Protegiendo a los animales de granja con inteligencia artificial — una herramienta al alcance de todo productor"* 🐑🐔🐹🛡️🦊
