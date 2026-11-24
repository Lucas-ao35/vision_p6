# Práctica de detección de objetos — Grupo N

**Integrantes:** Lucas Amado Oseira y María Lestón Geada  
**Dataset:** Lettuce Pallets (Roboflow-100, `Francesco/lettuce-pallets` en Hugging Face)  
**Detectores:** Cascade R-CNN R50-FPN (Detectron2) y SSD300-VGG16 (Torchvision), ambos preentrenados en MS COCO

## Contenido del zip
| Archivo | Descripción |
|---|---|
| `Grupo_N_practica_deteccion.ipynb` | Cuaderno completo y reproducible (baseline, mejoras, análisis) |
| `results.csv` | Tabla con todas las métricas de todos los experimentos (la genera el cuaderno) |
| `Grupo_N_deteccion.pdf` | Presentación de la defensa oral |
| `README.md` | Este archivo |

## Cómo ejecutarlo
1. Abrir el cuaderno en **Google Colab** con GPU (Entorno de ejecución → Cambiar tipo → GPU, p. ej. T4).
2. Ejecutar la celda de instalación (`pycocotools`, `ensemble-boxes`, `datasets`, `detectron2`). Si Colab pide reiniciar la sesión, reiniciar y seguir desde los imports.
3. Ejecutar **Runtime → Run all**. En la celda de configuración:
   - `DATA_SOURCE = "hf"` descarga el dataset de Hugging Face (opción por defecto, no requiere ficheros previos).
   - `USE_DRIVE = True` guarda checkpoints, predicciones y resultados en Google Drive (`DRIVE_WORK_DIR`), de modo que una desconexión no pierde el trabajo. Si ya existe el checkpoint de un experimento, se reutiliza y no se vuelve a entrenar (`force_retrain=True` lo fuerza).
4. Los experimentos se identifican por nombre (`ssd_baseline`, `ssd_E1_...`, `cascade_baseline`, ...). Cada uno queda registrado en `results.csv`.

## Reproducibilidad
- Semilla única `SEED = 2710` (Python, NumPy, PyTorch, workers de DataLoader, `cfg.SEED` de Detectron2); cuDNN en modo determinista.
- Las **mismas** funciones y umbrales (`SCORE_THRESH_EVAL`, `NMS_IOU`, `MAX_DETS`) evalúan ambos detectores.
- Protocolo: entrenar en `train`, seleccionar en `val`, reportar `test`.
- Versiones: _(COMPLETAR tras ejecutar: `torch`, `torchvision`, `detectron2`, `pycocotools`, GPU usada)_.

## Preprocesado de los datos
Se elimina la categoría padre `lettuce` (sin anotaciones), se recortan las cajas a la imagen, se descartan las cajas degeneradas y se renumeran las 5 clases reales a 1…5.

## Resultados principales
_(COMPLETAR: tabla resumida baseline vs mejores configuraciones, copiada de `results.csv`)_
