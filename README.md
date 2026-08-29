# KrishiDrishti AI — Image Diagnosis Module

Reproducible PyTorch/OpenCV image-diagnosis module for crop disease classification. It validates image quality before a trained MobileNetV3 classifier produces a diagnosis.

## Current experiment baseline

The initial dataset is Tomato PlantVillage, used only as a controlled baseline—not as a field-readiness benchmark. Experiment profiles are under `configs/experiments/`:

- `mobilenetv3_baseline.yaml`: CPU-oriented baseline.
- `efficientnet_b0.yaml`: stronger CNN comparison with moderate cost.
- `convnext_tiny.yaml`: higher-capacity CNN comparison.

Each profile extends `configs/default.yaml`, writes independent checkpoints/reports, and selects checkpoints by validation macro F1. Evaluation records accuracy, macro/weighted precision-recall-F1, per-class metrics, confusion matrix, expected calibration error, model size, and batch-1 latency.

## Dataset preparation: PlantVillage Tomato baseline

The current MVP uses only PlantVillage class directories that begin with `Tomato___`. The official repository layout is supported directly, including its leaf-grouping metadata:

```text
C:\Users\Vaibhav Sharma\OneDrive\Documents\New project\data\raw\PlantVillage\
  raw\color\Tomato___Bacterial_spot\
  raw\color\Tomato___Early_blight\
  raw\color\Tomato___healthy\
  leaf_grouping\leaf-map.json
```

Alternatively, pass the local root explicitly with `--dataset-dir`. The script only copies validated, unique source files. It never resizes or augments them.

When `leaf_grouping/leaf-map.json` is available, images sharing the same PlantVillage class-qualified leaf identifier are assigned as an indivisible group. The resulting image percentages may differ slightly from 70/15/15; group integrity takes priority. Without this metadata, the summary explicitly records `image_level_fallback`.

It creates `data/processed/{train,val,test}/Tomato___*/`, plus:

```text
artifacts/reports/class_mapping.json
artifacts/reports/dataset_summary.json
```

`dataset_summary.json` includes corrupt/unreadable files and duplicate hashes; duplicate images are excluded before splitting to prevent split leakage. Group-aware runs also report group counts, images-per-group statistics, split group/image counts, actual ratios, and confirmation that no group crosses splits.

### Usage

```powershell
python scripts\prepare_dataset.py --config configs\default.yaml --dataset-dir "C:\path\to\PlantVillage" --overwrite
```

With the configured default location:

```powershell
python scripts\prepare_dataset.py --config configs\default.yaml --overwrite
```

Run the test suite with `pytest`.

## First experiment command

After dataset preparation, run the controlled baseline experiment with:

```powershell
python scripts\train.py --config configs\experiments\mobilenetv3_baseline.yaml
```

Do not compare models using PlantVillage alone for deployment decisions. Add a farmer/field-image holdout before promoting any model.

## API

`POST /diagnose` accepts multipart form field `image`. A valid trained checkpoint is required; otherwise it returns HTTP 503 rather than fabricating a prediction.
