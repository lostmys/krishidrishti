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


## Farm analysis API

`POST /analyze-farm` is isolated from the crop-disease model pipeline. It accepts either a GeoJSON polygon or a point/radius pair and validates that the farm is inside India, the area exceeds 4 acres (~16,187.4 m²), and the date window does not exceed 30 days. A preferred 5-10 day window is used by default, and both the date window and abnormality thresholds remain configurable via request fields or environment variables.

Example polygon request:

```json
{
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[72.0, 18.0], [72.1, 18.0], [72.1, 18.1], [72.0, 18.1], [72.0, 18.0]]]
  },
  "date_window_days": 7,
  "percentile_threshold": 95.0,
  "zscore_threshold": 2.5
}
```

Example point/radius request:

```json
{
  "latitude": 19.0760,
  "longitude": 72.8777,
  "radius_meters": 1500,
  "date_window_days": 7
}
```

The endpoint uses service-account Earth Engine credentials created from environment variables without storing secrets in the repository:

```bash
export GEE_SERVICE_ACCOUNT_EMAIL="your-service-account@project.iam.gserviceaccount.com"
export GEE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
export GEE_PROJECT_ID="your-project-id"
export GEE_ANALYSIS_DATE_WINDOW_DAYS=7
export GEE_PERCENTILE_THRESHOLD=95
export GEE_ZSCORE_THRESHOLD=2.5
export GEE_MIN_CLUSTER_SIZE=3
```

The `/analyze-farm` response also includes a `region_image` data URL (PNG) generated from the actual satellite image crop for the supplied polygon or point/radius bounds. The image is clipped to the farm region and sized to fit the input geometry cleanly, so the returned image shows the relevant plot area rather than a placeholder rectangle.

The API also loads a `.env` file from the repository root automatically. Each setting must use `NAME=value` syntax, for example:

```dotenv
GEE_SERVICE_ACCOUNT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=/absolute/path/to/service-account-private-key.pem
GEE_PROJECT_ID=your-project-id
```

Do not commit `.env` or private keys. If a private key is exposed, revoke that key in Google Cloud and create a replacement.

The backend processes Sentinel-2 surface reflectance, applies cloud and shadow masking, and derives NDVI, NDMI, and NDRE. Sentinel-1 GRD VV/VH inputs supply VH/VV ratios and temporal change indicators. The farm-local anomaly score stays numerical and configurable. Each GeoJSON feature is a connected zone with `properties.label` (`anomaly`, `healthy`, or `unreachable`), `latitude`, `longitude`, and `radius_meters`, plus its pixel geometry and metrics. Position and radius are calculated from the sampled satellite pixel grid; they describe the zone's center and extent, while the geometry is the authoritative boundary. Cloud-masked/no-data pixels are labeled `unreachable`, not healthy.
