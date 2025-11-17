# MLflow + DVC Hybrid Architecture

**Document Version:** 1.0  
**Last Updated:** 2025-11-17  
**Team:** Equipo 56 - ITESM MNA MLOps

---

## Executive Summary

This document defines the target architecture for experiment tracking and artifact management in the Student Performance Prediction project. The solution implements a **hybrid approach** that separates concerns between MLflow (experiments) and DVC (data versioning).

---

## Architecture Overview

### Current Problem

- **MLflow** stores everything locally in `mlruns/` (~2GB+)
- Metadata and artifacts are not shared across team members
- Local paths make collaboration impossible
- DVC is underutilized (only tracking raw data)
- Inconsistent configuration across 6+ Python files

### Target Solution (Option C)

```
┌─────────────────────────────────────────────────────────────┐
│                    TEAM COLLABORATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Developer A (Local)          Developer B (Local)            │
│  ├─ mlflow.db (SQLite)        ├─ mlflow.db (SQLite)         │
│  └─ Tracks metadata           └─ Tracks metadata            │
│                                                               │
│          ↓                              ↓                     │
│          └──────────────┬───────────────┘                     │
│                         │                                     │
│                         ↓                                     │
│            ┌────────────────────────┐                         │
│            │   MLflow Server (Local) │                        │
│            │   Port: 5000            │                        │
│            └────────────────────────┘                         │
│                         │                                     │
│                         ↓                                     │
│            ┌────────────────────────┐                         │
│            │  S3 Artifact Storage    │                        │
│            │  s3://itesm-mna/        │                        │
│            │    202502-equipo56/     │                        │
│            │      mlflow/            │                        │
│            └────────────────────────┘                         │
│                                                               │
│            ┌────────────────────────┐                         │
│            │  DVC Remote Storage     │                        │
│            │  s3://itesm-mna/        │                        │
│            │    202502-equipo56/     │                        │
│            │      (root)             │                        │
│            └────────────────────────┘                         │
│                         ↑                                     │
│                         │                                     │
│            ┌────────────┴───────────┐                         │
│            │  DVC Tracked Files:     │                        │
│            │  • Raw data (.csv)      │                        │
│            │  • Production model     │                        │
│            │    (best_*.joblib)      │                        │
│            └────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. MLflow Tracking Server (Local)

**Technology:** MLflow Server with SQLite backend  
**Purpose:** Centralized experiment tracking with shared artifact storage

**Configuration:**
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://itesm-mna/202502-equipo56/mlflow \
  --host 0.0.0.0 \
  --port 5000
```

**What it does:**
- Stores experiment **metadata** locally (SQLite database)
- Stores experiment **artifacts** remotely (S3)
- Provides UI at `http://127.0.0.1:5000`
- Each team member runs their own server locally

**Key Benefits:**
- ✅ Metadata stays local (fast queries)
- ✅ Artifacts are shared (S3)
- ✅ No conflicts between team members
- ✅ Easy to query and compare experiments

### 2. S3 Artifact Storage

**Path:** `s3://itesm-mna/202502-equipo56/mlflow/`  
**Purpose:** Centralized storage for MLflow artifacts

**What gets stored here:**
- Trained models (per experiment run)
- Plots and visualizations
- Metrics and parameters (as artifacts)
- Model signatures and metadata

**Key Benefits:**
- ✅ Shared across team
- ✅ No git bloat
- ✅ Automatic versioning by MLflow

### 3. DVC (Data Version Control)

**Remote:** `s3://itesm-mna/202502-equipo56/` (root)  
**Purpose:** Version control for data and production models

**What DVC tracks:**
- `data/raw/*.csv` - Raw datasets
- `models/best_gridsearch_amplio.joblib` - Production model only

**What DVC does NOT track:**
- ❌ MLflow experiments (handled by MLflow)
- ❌ Intermediate artifacts (handled by MLflow)
- ❌ All experiment models (only production model)

**Key Benefits:**
- ✅ Reproducible data pipelines
- ✅ Production model versioning
- ✅ Separation of concerns

### 4. Centralized Configuration

**Module:** `mlops/mlflow_config.py`  
**Purpose:** Single source of truth for MLflow/AWS configuration

**What it provides:**
```python
# Configuration constants
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_S3_ARTIFACT_ROOT = "s3://itesm-mna/202502-equipo56/mlflow"
AWS_PROFILE = "equipo56"
AWS_REGION = "us-east-2"

# Setup function
setup_mlflow(experiment_name: str, run_name: Optional[str] = None)
```

**Key Benefits:**
- ✅ No configuration duplication
- ✅ Easy to update (one place)
- ✅ Environment variable overrides
- ✅ Automatic AWS credentials setup

---

## Data Flow

### Training Workflow

```
1. Developer starts MLflow server
   └─> mlflow server --backend-store-uri sqlite:///mlflow.db ...

2. Developer runs training script
   └─> python train/train_gridsearch.py

3. Script calls setup_mlflow()
   ├─> Configures AWS credentials
   ├─> Sets tracking URI: http://127.0.0.1:5000
   └─> Creates/sets experiment

4. During training:
   ├─> Metadata logged to local SQLite (mlflow.db)
   └─> Artifacts saved to S3 (s3://.../mlflow/...)

5. Developer views results
   └─> MLflow UI at http://127.0.0.1:5000
```

### Production Deployment

```
1. Select best model from MLflow UI

2. Export production model
   └─> joblib.dump(model, "models/best_gridsearch_amplio.joblib")

3. Track with DVC
   ├─> dvc add models/best_gridsearch_amplio.joblib
   ├─> git add models/best_gridsearch_amplio.joblib.dvc
   └─> git commit -m "Update production model"

4. Push to remote
   ├─> dvc push  # Model to S3
   └─> git push  # DVC pointer to Git

5. Other developers pull
   ├─> git pull  # Get DVC pointer
   └─> dvc pull  # Download model from S3
```

---

## File Structure

### What Goes Where

```
proyecto_mlops_equipo_56/
├── mlflow.db                    # Local (gitignored)
├── mlflow.db-journal            # Local (gitignored)
├── mlruns/                      # REMOVED (no longer used)
│
├── models/
│   ├── best_gridsearch_amplio.joblib      # DVC tracked
│   └── best_gridsearch_amplio.joblib.dvc  # Git tracked
│
├── data/
│   └── raw/
│       ├── student_entry_performance_original.csv      # DVC tracked
│       └── student_entry_performance_original.csv.dvc  # Git tracked
│
└── mlops/
    ├── mlflow_config.py         # Centralized config
    └── start_mlflow_server.sh   # Server startup script
```

### Git Tracking

**Tracked:**
- ✅ Source code (`.py`)
- ✅ DVC pointers (`.dvc`, `dvc.yaml`)
- ✅ Configuration (`mlflow_config.py`)
- ✅ Documentation (`.md`)

**NOT Tracked:**
- ❌ `mlflow.db*` (local metadata)
- ❌ `mlruns/` (deprecated)
- ❌ Actual data files (tracked by DVC)
- ❌ Actual model files (tracked by DVC)

---

## Key Decisions & Rationale

### Why Local SQLite + S3 Artifacts?

**Decision:** Use local SQLite for metadata, S3 for artifacts

**Rationale:**
- Metadata queries are fast (local)
- Artifacts are shared (S3)
- No need for remote database server
- Each developer has independent metadata view
- No conflicts when experimenting in parallel

### Why Not Track mlruns/ with DVC?

**Decision:** Do NOT use DVC to track `mlruns/`

**Rationale:**
- MLflow has better experiment tracking features
- DVC is not designed for thousands of small files
- Loses MLflow UI capabilities
- Hard to compare experiments
- S3 artifact storage is better suited

### Why Keep DVC for Production Model?

**Decision:** Use DVC only for final production model

**Rationale:**
- Production model needs strict versioning
- Separate from experimental models
- Easy to roll back in production
- Clear separation: experiments vs. production

### Why Centralize Configuration?

**Decision:** Single `mlflow_config.py` module

**Rationale:**
- Avoids configuration drift
- Easy to update tracking URI
- Consistent AWS credentials setup
- Easier to switch to remote MLflow server later

---

## Migration Path

### From Current → Target

**Step 1:** Archive existing `mlruns/`
```bash
mv mlruns ../mlruns_backup_$(date +%Y%m%d)
```

**Step 2:** Start MLflow server with new config
```bash
./mlops/start_mlflow_server.sh
```

**Step 3:** Run training with new setup
```bash
python train/train_gridsearch.py
```

**Step 4:** Verify artifacts in S3
- Check S3 console: `s3://itesm-mna/202502-equipo56/mlflow/`
- Check MLflow UI: `http://127.0.0.1:5000`

---

## Team Workflow

### Daily Development

1. **Start your MLflow server:**
   ```bash
   ./mlops/start_mlflow_server.sh
   ```

2. **Run experiments:**
   ```bash
   python train/train_gridsearch.py
   ```

3. **View results:**
   - Open: `http://127.0.0.1:5000`
   - All artifacts automatically in S3

4. **Compare with teammates:**
   - Everyone sees same artifacts (S3)
   - Metadata may differ (local SQLite)

### When Promoting to Production

1. **Identify best model in MLflow UI**

2. **Export to production path:**
   ```python
   joblib.dump(best_model, "models/best_gridsearch_amplio.joblib")
   ```

3. **Track with DVC:**
   ```bash
   dvc add models/best_gridsearch_amplio.joblib
   git add models/best_gridsearch_amplio.joblib.dvc
   git commit -m "Update production model v2.3"
   dvc push
   git push
   ```

4. **Teammates pull:**
   ```bash
   git pull
   dvc pull
   ```

---

## Environment Variables

### Required Configuration

```bash
# Option 1: Use AWS profile (recommended)
export AWS_PROFILE=equipo56
export AWS_DEFAULT_REGION=us-east-2

# Option 2: Direct credentials (for CI/CD)
export AWS_ACCESS_KEY_ID=AKIATAVAA573SWZYW3PI
export AWS_SECRET_ACCESS_KEY=<secret>
export AWS_DEFAULT_REGION=us-east-2

# MLflow tracking URI (optional override)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### AWS Profile Setup

**File:** `~/.aws/credentials`
```ini
[equipo56]
aws_access_key_id = AKIATAVAA573SWZYW3PI
aws_secret_access_key = <secret>
```

**File:** `~/.aws/config`
```ini
[profile equipo56]
region = us-east-2
output = json
```

---

## Troubleshooting

### Common Issues

**Issue:** "No module named 'boto3'"
```bash
pip install boto3 s3fs
```

**Issue:** "Access Denied" to S3
```bash
# Check AWS credentials
aws s3 ls s3://itesm-mna/202502-equipo56 --profile equipo56
```

**Issue:** MLflow UI shows no artifacts
```bash
# Verify S3 artifact root in server startup
./mlops/start_mlflow_server.sh
# Check logs for S3 connection errors
```

**Issue:** DVC push fails
```bash
# Verify DVC remote
dvc remote list
# Should show: team_remote  s3://itesm-mna/202502-equipo56
```

---

## Security Considerations

### Credentials Management

- ✅ Never commit AWS credentials to git
- ✅ Use AWS profiles or environment variables
- ✅ Rotate credentials periodically
- ✅ Limit S3 bucket access to team only

### S3 Access Control

- ✅ Bucket: `s3://itesm-mna/202502-equipo56`
- ✅ Profile: `equipo56` (read/write access)
- ✅ Region: `us-east-2`

---

## Future Enhancements

### Potential Improvements

1. **Remote MLflow Server**
   - Deploy MLflow to shared EC2/ECS instance
   - Replace `http://127.0.0.1:5000` with `http://mlflow-server.example.com`
   - No code changes needed (only `mlflow_config.py`)

2. **PostgreSQL Backend**
   - Replace SQLite with shared PostgreSQL
   - Better for concurrent access
   - Shared metadata across team

3. **Model Registry**
   - Use MLflow Model Registry
   - Stage transitions (Staging → Production)
   - Approval workflows

4. **CI/CD Integration**
   - Automated model training on PRs
   - Automatic model validation
   - Production deployment pipelines

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)
- Project Repository: `proyecto_mlops_equipo_56`

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-17 | Initial architecture design |

---

**End of Document**
