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
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Farmer Web UI & Extension Portal                      │
│   Crop selector · Leaf photo upload · Farm polygon/coordinates · Language   │
│   Multilingual diagnosis · Field satellite screening · Case incident report │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP / REST API
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                       FastAPI Application (api.py)                          │
│   Canonical entrypoint · Validation · Upload handling · Multilingual router │
└──────────┬───────────────────┬───────────────────┬───────────────────┬──────┘
           │                   │                   │                   │
┌──────────▼──────────┐ ┌──────▼──────────┐ ┌──────▼──────────┐ ┌──────▼──────────┐
│   Leaf Diagnosis    │ │ Satellite Module│ │   Risk Fusion   │ │ Case & Workflow  │
│  Inference Pipeline │ │  (GEE Adapter)  │ │     Engine      │ │    Management    │
├─────────────────────┤ ├─────────────────┤ ├─────────────────┤ ├──────────────────┤
│ • Quality Gate      │ │ • Sentinel-2    │ │ • Leaf AI (40%) │ │ • SQLite Store   │
│ • CropRegistry      │ │   NDVI/NDMI/NDRE│ │ • Satellite(35%)│ │ • State Machine  │
│ • MobileNetV3-Small │ │ • Sentinel-1 SAR│ │ • Outbreak (15%)│ │ • Expert Review  │
│ • Calibrated Conf   │ │ • Anomaly Detect│ │ • Urgency (10%) │ │ • 48h Follow-up  │
│ • ICAR Knowledge    │ │ • Graceful degr.│ │ • Deterministic │ │ • Notifications  │
└─────────────────────┘ └─────────────────┘ └─────────────────┘ └──────────────────┘
```

**Key architectural principles:**
1. **Separation of Concerns & Modality Roles:** Satellite screening detects field-scale anomalies (vigor decline, moisture stress), while leaf computer vision diagnoses specific pathogens on plant leaves.
2. **Data-Driven Crop Routing:** Adding a new crop requires configuration in `configs/default.yaml`, an experiment YAML, ICAR knowledge base entries, and model weights — zero API code changes.
3. **Resilient Degradation:** Missing Earth Engine credentials or offline connectivity do not crash leaf diagnosis or case creation workflows.
4. **Deterministic Risk Scoring:** Multi-factor risk calculation is fully transparent, auditable, and rule-based (no ungrounded LLM calculations).
5. **Human-in-the-Loop:** Cases flagged as high-risk or uncertain automatically enter expert review queues for KVK/Agronomist verification.

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

### Diagnosis & Core Services
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Farmer web UI |
| `GET` | `/health` | System readiness status for crops, GEE, and SQLite DB |
| `GET` | `/crops` | List of registered crop slugs (tomato, soyabean, cotton) |
| `POST` | `/diagnose/{crop}?language=en` | Crop-specific leaf diagnosis with multilingual ICAR advice |
| `POST` | `/diagnose` | Single-model diagnosis endpoint (backward compatibility) |
| `POST` | `/analyze-farm` | Direct Google Earth Engine satellite farm screening |

### Incident Reporting & Case Management
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/cases` | Create farmer case with optional leaf photo, coordinates, and symptoms |
| `GET` | `/cases` | Filter cases by `status`, `crop`, `risk_level`, or `village` |
| `GET` | `/cases/due-followup` | Identify active cases requiring 48-hour health follow-up |
| `GET` | `/cases/{case_id}?language=mr` | Get case details, timeline, reviews, and localized guidance |
| `POST` | `/cases/{case_id}/satellite` | Trigger or refresh satellite screening for a case polygon |
| `POST` | `/cases/{case_id}/review` | Agronomist / KVK expert review (`CONFIRM`, `REJECT`, `REQUEST_FIELD_VISIT`, `NOTE`) |
| `POST` | `/cases/{case_id}/follow-up` | Farmer condition check (`IMPROVED`, `SAME`, `WORSE`) |

### Supported Languages

| Code | Language |
|------|----------|
| `en` | English (default) |
| `hi` | Hindi (हिन्दी) |
| `mr` | Marathi (मराठी) |

---

## Multi-Source Risk Fusion Engine

KrishiDrishti integrates leaf-level vision and satellite-level remote sensing into a transparent, deterministic composite risk score:

$$Risk = 0.40 \cdot S_{\text{leaf}} + 0.35 \cdot S_{\text{satellite}} + 0.15 \cdot S_{\text{outbreak}} + 0.10 \cdot S_{\text{urgency}}$$

1. **Leaf Disease Score ($0.40$):** Calibrated classifier confidence weighted by pathogen lethality (e.g., Late Blight vs Healthy).
2. **Satellite Anomaly Score ($0.35$):** Sentinel-2 vegetation index decline (NDVI, NDMI moisture stress, NDRE canopy chlorophyll) and Sentinel-1 SAR backscatter changes. Missing satellite data gracefully falls back to neutral without skewing.
3. **Outbreak Proximity ($0.15$):** Recent confirmed disease cases reported in the same village or district.
4. **Farmer Symptom Urgency ($0.10$):** Rapid progression or high percentage of crop area affected.

### Risk Levels & Operational Action

| Score Range | Category | Workflow Action |
|-------------|----------|-----------------|
| $\ge 0.75$ | `CRITICAL` | High-priority field visit requested; immediate automated SMS alert |
| $0.50 - 0.74$ | `HIGH` | Queued for agronomist expert review within 24 hours |
| $0.30 - 0.49$ | `MODERATE` | Standard multilingual guidance dispatched to farmer |
| $< 0.30$ | `LOW` | Preventive recommendations and scheduled 48h follow-up |

> **Notice:** The composite risk score is a research decision-support tool. It does not supersede official ICAR-KVK agricultural directives.

---

## Case Lifecycle & Expert Review

```
                ┌──────────────┐
                │  POST /cases │
                └──────┬───────┘
                       │
                 ┌─────▼────┐
           ┌─────┤   OPEN   ├────────┐
           │     └─────┬────┘        │
(Low Conf /│           │ (High Conf/ │ (Expert
 High Risk)│           │  Normal)    │  Decline)
     ┌─────▼──────────┐│             ▼
     │REVIEW_         ││         ┌──────────┐
     │RECOMMENDED     ││         │ REJECTED │
     └─────┬──────────┘│         └──────────┘
           │           │
           │ (Expert   │ (Expert Confirm)
           │  Confirm) │
           │     ┌─────▼─────┐
           └────►│ CONFIRMED │
                 └─────┬─────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌──────────────────────┐   ┌────────────────────────┐
│ FIELD_VISIT_REQUIRED │   │   48h Follow-up Flow   │
└──────────────────────┘   │ • IMPROVED ──► RESOLVED│
                           │ • SAME     ──► Ongoing │
                           │ • WORSE    ──► Escalated│
                           └────────────────────────┘
```

---

## Google Earth Engine (GEE) Setup

The satellite module operates safely in **both connected and disconnected environments**:

### Configuration (`.env`)
```bash
GEE_SERVICE_ACCOUNT=krishidrishti-sa@your-gcp-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_FILE=configs/credentials/gee_key.json
```

- **When Credentials Exist:** Full Sentinel-2 optical & Sentinel-1 SAR analysis runs against the farmer's geometry.
- **When Credentials Are Absent:** The service returns a clean degradation status (`configured: false`), logging an advisory without throwing 500 errors or disrupting leaf diagnosis.

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

1. **Active Crop Models.** Tomato (10 classes), Soyabean (5 classes), and Cotton configured. Additional crops can be added purely through configuration and checkpoints.
2. **Lab-image training data.** Models were trained on PlantVillage (Tomato) and curated datasets (Soyabean), not field-level Maharashtra camera photos. Real-world field accuracy may differ.
3. **No authentication.** API endpoints are unauthenticated — suitable for internal demo/pilot use.
4. **Google Earth Engine dependency.** Advanced satellite indices require valid GCP/GEE service account credentials; when absent, the system operates in local degraded mode.
5. **Single-worker deployment.** Each uvicorn worker loads its own model copies (~6 MB each). For production scale, consider model serving infrastructure (Triton/TorchServe).
6. **No offline mobile client.** The web UI requires network connectivity to the API server.

---

## Testing

```powershell
.venv\Scripts\pytest
```

The test suite (71 tests across 13 test suites) covers:
- Configuration inheritance and path resolution
- Image quality gate (blur, resolution, brightness)
- Confidence status logic and temperature calibration
- Label parsing (Tomato `___` separator, Soyabean, Cotton)
- CropRegistry (lazy loading, caching, unknown slug errors, case normalization)
- API routing (health, crops, diagnose, error codes 400/404/415/503)
- Upload size limit enforcement
- Disease knowledge lookups (Tomato, Soyabean, Cotton, Healthy, unknown diagnosis)
- Multilingual responses (English, Hindi, Marathi)
- Multi-Source Risk Fusion engine (weighting, edge cases, explanations)
- Resilient Satellite screening & GEE adapter fallbacks
- SQLite Case Management & Expert Review state machine lifecycle
- Simulated Multilingual Notification dispatching
- ECE and classification metrics computation

---

## Project Structure

```
configs/
  default.yaml                          # Base config + crop registry (Tomato, Soyabean, Cotton)
  experiments/
    tomato_mobilenetv3.yaml             # Tomato experiment config
    soyabean_mobilenetv3.yaml           # Soyabean experiment config
data/
  krishidrishti.db                      # SQLite Case & Review Database
  raw/PlantVillage/                     # Raw Tomato images
  raw/Soyabean/                         # Raw Soyabean images
  processed/tomato/{train,val,test}/    # Processed Tomato splits
  processed/soyabean/{train,val,test}/  # Processed Soyabean splits
  knowledge/disease_knowledge.yaml      # Multilingual ICAR disease knowledge base
  uploads/                              # Case management leaf photo storage
artifacts/
  checkpoints/tomato/mobilenetv3_baseline.pt
  checkpoints/soyabean/mobilenetv3_baseline.pt
  reports/tomato/mobilenetv3_baseline/  # Tomato evaluation results
  reports/soyabean/mobilenetv3_baseline/ # Soyabean evaluation results
src/krishidrishti_ai/
  api.py                                # Canonical FastAPI application & router
  api_vaibhav.py                        # Backward-compatible wrapper
  config.py                             # YAML config loader with inheritance
  metrics.py                            # ECE and classification metrics
  services/
    cases.py                            # SQLite-backed CaseService & state machine
    fusion.py                           # RiskFusionService (Leaf + Satellite + Outbreak)
    inference.py                        # DiagnosisService (quality + model + knowledge)
    knowledge.py                        # DiseaseKnowledgeService (ICAR multilingual)
    notifications.py                    # NotificationProvider & LocalNotificationAdapter
    quality.py                          # Image quality assessment
    registry.py                         # CropRegistry (data-driven routing)
    satellite.py                        # Safe GEE SatelliteService adapter
  models/classifier.py                  # Model builder (MobileNetV3, EfficientNet, etc.)
  data/transforms.py                    # Image transforms
  static/index.html                     # Farmer-facing web UI
scripts/
  prepare_dataset.py                    # Dataset preparation
  train.py                             # Training loop
  evaluate.py                          # Evaluation pipeline
tests/
  test_api.py                          # API endpoint tests
  test_calibration.py                  # Temperature scaling tests
  test_case_management.py              # Case lifecycle & follow-up tests
  test_config.py                       # Config loading tests
  test_cotton.py                       # Cotton crop routing & knowledge tests
  test_fusion.py                       # Risk fusion engine tests
  test_gee_api.py                      # Earth engine adapter & resilience tests
  test_inference_logic.py              # Inference & registry tests
  test_knowledge.py                    # Multilingual knowledge layer tests
  test_metrics.py                      # Metrics computation tests
  test_notifications.py                # Notification dispatch tests
  test_prepare_dataset.py              # Dataset preparation tests
  test_quality.py                      # Image quality gate tests
```
