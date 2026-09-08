# LUNA-MATCH Core Engine

LUNA-MATCH is a high-precision lunar image co-registration engine engineered for ISRO Chandrayaan-2 multi-sensor datasets (including OHRC, TMC-2, and IIRS) referenced against LROC NAC basemaps. It combines illumination-invariant Phase Congruency, Maximum Intensity Moments (MIM), Classical SIFT with Adaptive Non-Maximal Suppression (ANMS), MAGSAC++ robust geometric estimation with relief-significance testing, and Lucas-Kanade gradient refinement to deliver sub-pixel alignment accuracy under extreme lighting variations, terrain relief, and scale disparities.

---

## Setup (Tested & Confirmed)

Create a fresh virtual environment using Python 3.12 and install the core dependencies:

```bash
python3 -m venv .venv_fresh
./.venv_fresh/bin/pip install --upgrade pip
./.venv_fresh/bin/pip install -r requirements.txt
```

---

## Running Verification & Tests

Execute the comprehensive 74-test verification suite:

```bash
./.venv_fresh/bin/pytest tests/ -v
```

---

## Running the Verification Demo

Run the standalone end-to-end registration demonstration on calibrated Chandrayaan-2 / LROC sample rasters:

```bash
./.venv_fresh/bin/python demo.py --img-a data/samples/verified_a.tif --img-b data/samples/verified_b.tif --out demo_output/fresh_check
```

Expected output:
- Registration completed in ~1.2 seconds.
- Mean Reprojection RMSE: ~0.33 px (< 0.5 px sub-pixel threshold).
- Inlier Ratio: ~89.2% (514 verified correspondences).

---

## Running API Locally

Launch the high-performance asynchronous FastAPI service using Uvicorn:

```bash
./.venv_fresh/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation will be accessible at `http://localhost:8000/docs`.

---

## Docker Deployment

Build and run the containerized engine:

```bash
docker build -t luna-match-engine .
docker run -d -p 8000:8000 luna-match-engine
```

Verify health:
```bash
curl -s http://localhost:8000/health
```

---

## Project Structure

- `core/`: Algorithmic modules (raster I/O, Lommel-Seeliger normalization, Phase Congruency, ANMS, MAGSAC++, TPS, Lucas-Kanade refinement, and metrics calculation).
- `pipeline/`: Orchestration state machine (`LunaMatchPipeline`) managing lifecycle stages and job state transitions.
- `api/`: FastAPI server implementation (`main.py`), Pydantic models (`schemas.py`), and lifecycle endpoints.
- `tests/`: Automated test suite covering all mathematical kernels, pipeline state transitions, and API endpoints (74 tests).
- `data/`: Sample rasters (`data/samples/`) and persistent job workspace storage (`data/jobs/`).
- `scripts/`: Dataset preparation, ground-truth verification, and illumination stress test generators.

---

## API Documentation & Contract

For endpoint schemas, multipart upload details, and the unified Chatbot/VLM/RAG response specification, see [API_CONTRACT.md](API_CONTRACT.md).
