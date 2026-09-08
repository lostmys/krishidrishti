# KrishiDrishti AI — Crop Disease Diagnosis System

AI-powered crop disease diagnosis system targeted at Maharashtra farmers. A farmer uploads a leaf photo, and the system identifies the disease, explains symptoms, and provides actionable guidance in English, Hindi, or Marathi.

> **⚠ Safety Notice:** This system provides preliminary AI screening only. It does not replace professional agricultural advice. Always consult your local Krishi Vigyan Kendra (KVK) or Agricultural Extension Officer before applying treatment.

---

## Supported Crops & Classes

### Tomato (10 classes)

| # | Class | Test F1 |
|---|-------|---------|
| 1 | Bacterial Spot | 98.6% |
| 2 | Early Blight | 94.1% |
| 3 | Healthy | 99.6% |
| 4 | Late Blight | 97.2% |
| 5 | Leaf Mold | 96.8% |
| 6 | Septoria Leaf Spot | 97.9% |
| 7 | Spider Mites (Two-Spotted) | 97.2% |
| 8 | Target Spot | 98.6% |
| 9 | Tomato Mosaic Virus | 94.9% |
| 10 | Tomato Yellow Leaf Curl Virus | 98.6% |

**Aggregate:** Accuracy 97.91% · Macro F1 97.35% · ECE 0.0055

### Soyabean (5 classes)

| # | Class | Test F1 |
|---|-------|---------|
| 1 | Bacterial Blight | 100.0% |
| 2 | Cercospora Leaf Blight | 87.0% |
| 3 | Healthy | 96.3% |
| 4 | Rust | 93.3% |
| 5 | Sudden Death Syndrome | 100.0% |

**Aggregate:** Accuracy 95.65% · Macro F1 95.32% · ECE 0.0305

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Farmer Web UI (index.html)               │
│   Crop selector · Photo upload · Language selector  │
│   Result card · Symptoms · Actions · Prevention     │
└─────────────────────┬───────────────────────────────┘
                      │ POST /diagnose/{crop}?language=mr
┌─────────────────────▼───────────────────────────────┐
│                 FastAPI (api.py)                     │
│   Routes · Upload validation · Error handling       │
└─────────────┬───────────────┬───────────────────────┘
              │               │
    ┌─────────▼─────┐  ┌─────▼──────────────────┐
    │ CropRegistry  │  │ DiseaseKnowledgeService │
    │ (registry.py) │  │ (knowledge.py)          │
    │ slug → config │  │ YAML → multilingual     │
    │ lazy-load     │  │ advice lookup           │
    └───────┬───────┘  └─────────────────────────┘
            │
  ┌─────────▼──────────┐
  │  DiagnosisService   │
  │  (inference.py)     │
  │  Quality gate →     │
  │  MobileNetV3 →      │
  │  Top-k predictions  │
  └─────────────────────┘
```

**Key design principle:** Crop routing is entirely data-driven. Adding a new Maharashtra crop (Cotton, Onion, Sugarcane, etc.) requires:
1. One new line in `configs/default.yaml` under `crops:`
2. One new experiment config YAML
3. Disease entries in `data/knowledge/disease_knowledge.yaml`
4. A trained checkpoint

No Python code changes needed.

---

## Installation

```powershell
# Clone and enter the project
cd "path/to/project"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch, torchvision
- FastAPI, uvicorn
- OpenCV, Pillow
- scikit-learn, matplotlib, seaborn
- PyYAML

---

## Dataset Preparation

### Tomato (PlantVillage)

```powershell
python scripts\prepare_dataset.py --config configs\experiments\tomato_mobilenetv3.yaml --overwrite
```

Expects raw data at `data/raw/PlantVillage/`. Supports group-aware splitting via `leaf-map.json`.

### Soyabean

```powershell
python scripts\prepare_dataset.py --config configs\experiments\soyabean_mobilenetv3.yaml --overwrite
```

Expects raw data at `data/raw/Soyabean/` with class subdirectories.

Processed datasets are written to `data/processed/tomato/` and `data/processed/soyabean/` respectively, with `{train,val,test}/{class_name}/` structure.

---

## Training

```powershell
# Tomato
python scripts\train.py --config configs\experiments\tomato_mobilenetv3.yaml

# Soyabean
python scripts\train.py --config configs\experiments\soyabean_mobilenetv3.yaml
```

Checkpoints are saved to `artifacts/checkpoints/{crop}/mobilenetv3_baseline.pt`. Training selects the best checkpoint by validation macro F1.

---

## Evaluation

```powershell
# Tomato
python scripts\evaluate.py --config configs\experiments\tomato_mobilenetv3.yaml

# Soyabean
python scripts\evaluate.py --config configs\experiments\soyabean_mobilenetv3.yaml
```

Reports are written to `artifacts/reports/{crop}/mobilenetv3_baseline/`:
- `metrics.json` — accuracy, macro/weighted P/R/F1, ECE, latency, model size
- `per_class_report.json` — per-class precision, recall, F1, support
- `confusion_matrix.png` — labeled confusion matrix

---

## Starting the API

```powershell
.venv\Scripts\uvicorn krishidrishti_ai.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in a browser to access the farmer-facing web UI.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Farmer web UI |
| `GET` | `/health` | Readiness status for all registered crops |
| `GET` | `/crops` | List of registered crop slugs |
| `POST` | `/diagnose/{crop}?language=en` | Crop-specific diagnosis with multilingual advice |
| `POST` | `/diagnose` | Legacy single-model endpoint (backward compat) |

### Supported languages

| Code | Language |
|------|----------|
| `en` | English (default) |
| `hi` | Hindi (हिन्दी) |
| `mr` | Marathi (मराठी) |

### Farm-analysis image output

Farm-analysis PNGs are saved locally instead of embedded as base64 in the JSON response.
The response includes the absolute path in `region_image_path`. By default, files are
written to `artifacts/gee_images/`; override this with:

```env
GEE_IMAGE_OUTPUT_DIR=/path/to/farm-images
```

The farm response also includes `farm` (the submitted polygon or a derived circle),
`label` (`Healthy`, `Abnormal`, or `Critical`), `mean_metrics`, and `anomalies`.
Each anomaly contains its latitude, longitude, score, and grid location.

---

## Example Diagnosis Request

```bash
curl -X POST "http://127.0.0.1:8000/diagnose/soyabean?language=mr" \
     -F "image=@leaf.jpg"
```

### Example Response

```json
{
  "crop": "Soyabean",
  "diagnosis": "Bacterial Blight",
  "confidence": 0.999893,
  "image_quality": {
    "passed": true,
    "blur_score": 160.97,
    "brightness_score": 98.86
  },
  "status": "AI_CONFIDENT",
  "top_predictions": [
    {"crop": "Soyabean", "diagnosis": "Bacterial Blight", "confidence": 0.999893},
    {"crop": "Soyabean", "diagnosis": "Sudden Death Syndrome", "confidence": 0.000081},
    {"crop": "Soyabean", "diagnosis": "Healthy", "confidence": 0.000018}
  ],
  "advice": {
    "display_name": "सोयाबीन जिवाणू करपा (Bacterial Blight)",
    "description": "सुडोमोनस जिवाणूंमुळे होतो...",
    "symptoms": ["पानांवर पिवळी कडा असलेले पाण्याचे ठिपके.", "..."],
    "immediate_actions": ["पाने ओली असताना शेतात फिरणे टाळा.", "..."],
    "prevention": ["प्रमाणित बियाण्यांची पेरणी करा.", "..."],
    "severity_guidance": "कमी ते मध्यम.",
    "source": "ICAR - Indian Institute of Soybean Research (IISR), Indore"
  }
}
```

---

## Confidence & Status Meanings

| Status | Condition | Behavior |
|--------|-----------|----------|
| `AI_CONFIDENT` | confidence ≥ 0.75 | Full diagnosis with disease-specific advice |
| `REVIEW_RECOMMENDED` | 0.45 ≤ confidence < 0.75 | Diagnosis shown with warning to verify with expert |
| `LOW_CONFIDENCE` | confidence < 0.45 | No disease-specific advice; recommends retaking photo or consulting expert |
| `IMAGE_QUALITY_REJECTED` | Image fails quality gate | No prediction; guidance on how to take a better photo |

### Image Quality Gate

Images must pass minimum thresholds for resolution (128×128), blur (Laplacian variance ≥ 80), and brightness (35–220 mean gray value).

---

## Safety & Limitations

### Safety boundaries

- **No chemical pesticide prescriptions.** The system recommends consulting agricultural extension officers for treatment decisions.
- **No fabricated advice.** All disease knowledge is curated from ICAR research institute guidelines, not generated by an LLM.
- **Uncertain predictions are flagged.** Low-confidence results explicitly recommend expert verification rather than presenting uncertain diagnoses as fact.

### Current MVP limitations

1. **Two crops only.** Tomato (10 classes) and Soyabean (5 classes). Additional Maharashtra crops can be added via configuration.
2. **Lab-image training data.** Models were trained on PlantVillage (Tomato) and curated web datasets (Soyabean), not Maharashtra field photos. Real-world accuracy may differ.
3. **No authentication.** API endpoints are unauthenticated — suitable for internal demo/pilot use.
4. **No satellite module.** Satellite-based crop monitoring is out of scope for this MVP.
5. **Single-worker deployment.** Each uvicorn worker loads its own model copies (~6 MB each). For production scale, consider a model serving infrastructure (Triton/TorchServe).
6. **No offline mode.** The farmer UI requires network connectivity to the API server.

---

## Testing

```powershell
.venv\Scripts\pytest
```

The test suite covers:
- Configuration inheritance and path resolution
- Image quality gate (blur, resolution)
- Confidence status logic
- Label parsing (Tomato `___` separator and Soyabean plain labels)
- CropRegistry (lazy loading, caching, unknown slug errors, case normalization)
- API routing (health, crops, diagnose, error codes 400/404/415/503)
- Upload size limit enforcement
- Disease knowledge lookups (Tomato, Soyabean, Healthy, unknown diagnosis)
- Multilingual responses (English, Hindi, Marathi)
- Confidence safety handling (AI_CONFIDENT, REVIEW_RECOMMENDED, LOW_CONFIDENCE, IMAGE_QUALITY_REJECTED)
- ECE and classification metrics computation

---

## Project Structure

```
configs/
  default.yaml                          # Base config + crop registry
  experiments/
    tomato_mobilenetv3.yaml             # Tomato experiment config
    soyabean_mobilenetv3.yaml           # Soyabean experiment config
data/
  raw/PlantVillage/                     # Raw Tomato images
  raw/Soyabean/                         # Raw Soyabean images
  processed/tomato/{train,val,test}/    # Processed Tomato splits
  processed/soyabean/{train,val,test}/  # Processed Soyabean splits
  knowledge/disease_knowledge.yaml      # Multilingual disease knowledge base
artifacts/
  checkpoints/tomato/mobilenetv3_baseline.pt
  checkpoints/soyabean/mobilenetv3_baseline.pt
  reports/tomato/mobilenetv3_baseline/  # Tomato evaluation results
  reports/soyabean/mobilenetv3_baseline/ # Soyabean evaluation results
src/krishidrishti_ai/
  api.py                                # FastAPI application
  config.py                             # YAML config loader with inheritance
  metrics.py                            # ECE and classification metrics
  services/
    inference.py                        # DiagnosisService (quality + model + knowledge)
    knowledge.py                        # DiseaseKnowledgeService
    quality.py                          # Image quality assessment
    registry.py                         # CropRegistry (data-driven routing)
  models/classifier.py                  # Model builder (MobileNetV3, EfficientNet, etc.)
  data/transforms.py                    # Image transforms
  static/index.html                     # Farmer-facing web UI
scripts/
  prepare_dataset.py                    # Dataset preparation
  train.py                             # Training loop
  evaluate.py                          # Evaluation pipeline
tests/
  test_api.py                          # API endpoint tests
  test_config.py                       # Config loading tests
  test_inference_logic.py              # Inference & registry tests
  test_knowledge.py                    # Knowledge layer tests
  test_metrics.py                      # Metrics computation tests
  test_prepare_dataset.py             # Dataset preparation tests
  test_quality.py                      # Image quality gate tests
```
