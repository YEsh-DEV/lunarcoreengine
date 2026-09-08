"""
api/main.py
===========
FastAPI application for LUNA-MATCH Planetary Image Registration Workbench.

Exposes core endpoints:
  - GET  /health           : Health check and diagnostic status
  - POST /register          : Submit a registration job
  - GET  /jobs/{id}         : Query job lifecycle status
  - GET  /jobs/{id}/result  : Retrieve quantitative registration metrics & artifacts
  - GET  /jobs/{id}/preview : Fetch raster or residual preview visualization
"""

import os
import base64
import json
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import LunaMatchPipeline, JobState
from collections import defaultdict
from api.schemas import RegisterRequest, JobStatusResponse, JobResultResponse, ChatbotSummaryResponse, ChatRequest, ChatResponse, ChatHistoryResponse, AnalyzeResponse, OrchestrateRequest, OrchestrateResponse
from core.ingest_preprocess import read_raster
from core.summary_builder import build_chatbot_summary

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

logger = logging.getLogger("luna-match")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup checks
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — /chat and /analyze will return 503")
    else:
        logger.info("GROQ_API_KEY configured — chat endpoints active")
    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    logger.info(f"Using GROQ_MODEL: {model}")
    logger.info("CORS enabled: allow_origins=['*'], allow_methods=['*'], allow_headers=['*'] (open to all domains)")
    yield


app = FastAPI(title="LUNA-MATCH", lifespan=lifespan)

# Enable CORS for frontend workbench integration
# In-memory session store for multi-turn chat
chat_sessions: dict[str, list[dict]] = defaultdict(list)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://luna-match-sih.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, tile_size: int = 36) -> np.ndarray:
    """Create a checkerboard mosaic interleaving img1 and img2 in blocks of tile_size x tile_size."""
    H, W = img1.shape[:2]
    if img2.shape[:2] != (H, W):
        img2 = cv2.resize(img2, (W, H))

    checkerboard = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y_end = min(H, y + tile_size)
            x_end = min(W, x + tile_size)

            tile_y = (y // tile_size) % 2
            tile_x = (x // tile_size) % 2

            if (tile_y + tile_x) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def draw_tie_points(
    img_a: np.ndarray,
    img_b: np.ndarray,
    matches: np.ndarray,
    max_draw: int = 80,
) -> np.ndarray:
    """Draw side-by-side correspondence lines connecting matched keypoints."""
    H_a, W_a = img_a.shape[:2]
    H_b, W_b = img_b.shape[:2]

    canvas_h = max(H_a, H_b)
    canvas_w = W_a + W_b

    def to_u8(img):
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    c_a = to_u8(img_a)
    c_b = to_u8(img_b)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:H_a, :W_a] = cv2.cvtColor(c_a, cv2.COLOR_GRAY2BGR)
    canvas[:H_b, W_a:canvas_w] = cv2.cvtColor(c_b, cv2.COLOR_GRAY2BGR)

    if len(matches) > max_draw:
        step = len(matches) // max_draw
        draw_matches = matches[::step][:max_draw]
    else:
        draw_matches = matches

    for m in draw_matches:
        x1, y1, x2, y2 = int(round(m[0])), int(round(m[1])), int(round(m[2])), int(round(m[3]))
        pt1 = (x1, y1)
        pt2 = (x2 + W_a, y2)
        conf = float(m[4]) if len(m) > 4 else 0.8

        color = (0, 240, 100) if conf > 0.6 else (255, 200, 0)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    return canvas


def _find_job_inputs(job_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    paths_file = job_dir / "input" / "input_paths.json"
    if paths_file.exists():
        try:
            with open(paths_file, "r") as f:
                d = json.load(f)
                if d.get("img_a_path") and d.get("img_b_path"):
                    return d["img_a_path"], d["img_b_path"]
        except Exception:
            pass

    meta_file = job_dir / "input" / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                d = json.load(f)
                if "img_a_path" in d and "img_b_path" in d:
                    return d["img_a_path"], d["img_b_path"]
        except Exception:
            pass

    return None, None


@app.get("/health")
def health_check():
    """Service health and diagnostic status."""
    return {
        "status": "ok",
        "service": "LUNA-MATCH Registration API",
        "version": "2.0.0",
        "default_matcher": "classical",
        "structural_matching_available": True,
        "active_workers": _executor._max_workers,
    }


def _execute_pipeline_task(job_id: str, img_a_path: str, img_b_path: str):
    """Background runner for LunaMatchPipeline."""
    try:
        pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path)
        result = pipeline.run()
        logger.info(f"Job {job_id} finished with status: {result.get('status')}")
    except Exception as e:
        logger.error(f"Execution error for job {job_id}: {e}")


@app.post("/register", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def register_images(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Initiate a new registration pipeline run between Chandrayaan-2 moving imagery
    and reference lunar imagery. Accepts JSON payload or multipart form data.
    """
    content_type = request.headers.get("content-type", "")
    job_id = None
    img_a_path = None
    img_b_path = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        job_id = form.get("job_id")
        if not job_id:
            job_id = f"job_{uuid.uuid4().hex[:10]}"

        job_input_dir = Path("data") / "jobs" / job_id / "input"
        job_input_dir.mkdir(parents=True, exist_ok=True)

        file_a = form.get("img_a")
        file_b = form.get("img_b")

        if file_a is not None and hasattr(file_a, "filename") and file_a.filename:
            path_a = job_input_dir / f"upload_a_{file_a.filename}"
            content = await file_a.read()
            with open(path_a, "wb") as f_out:
                f_out.write(content)
            img_a_path = str(path_a)
        elif "img_a_path" in form:
            img_a_path = str(form.get("img_a_path"))

        if file_b is not None and hasattr(file_b, "filename") and file_b.filename:
            path_b = job_input_dir / f"upload_b_{file_b.filename}"
            content = await file_b.read()
            with open(path_b, "wb") as f_out:
                f_out.write(content)
            img_b_path = str(path_b)
        elif "img_b_path" in form:
            img_b_path = str(form.get("img_b_path"))

    else:
        try:
            body = await request.json()
            req = RegisterRequest(**body)
            job_id = req.job_id or f"job_{uuid.uuid4().hex[:10]}"
            img_a_path = req.img_a_path
            img_b_path = req.img_b_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid registration request: {e}")

    if not img_a_path:
        raise HTTPException(status_code=400, detail="Missing source image (img_a or img_a_path)")
    if not img_b_path:
        raise HTTPException(status_code=400, detail="Missing reference image (img_b or img_b_path)")

    if not os.path.exists(img_a_path):
        raise HTTPException(status_code=400, detail=f"Source image not found: {img_a_path}")
    if not os.path.exists(img_b_path):
        raise HTTPException(status_code=400, detail=f"Reference image not found: {img_b_path}")

    # Initialize job directory structure and PENDING status
    pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path)

    # Launch execution asynchronously
    background_tasks.add_task(_execute_pipeline_task, job_id, img_a_path, img_b_path)

    return JobStatusResponse(
        job_id=job_id,
        status=JobState.PENDING.value,
        elapsed_s=0.0,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Query current execution state and elapsed runtime of a registration job."""
    status_file = Path("data") / "jobs" / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job ID not found: {job_id}")

    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        return JobStatusResponse(
            job_id=data.get("job_id", job_id),
            status=data.get("status", "UNKNOWN"),
            elapsed_s=data.get("elapsed_s", 0.0),
            error=data.get("error", None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse job status: {e}")


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    """Retrieve quantitative validation metrics and output files for a completed job."""
    job_dir = Path("data") / "jobs" / job_id
    status_file = job_dir / "status.json"
    metrics_file = job_dir / "output" / "metrics.json"

    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job ID not found: {job_id}")

    with open(status_file, "r") as f:
        status_data = json.load(f)

    st = status_data.get("status", "UNKNOWN")

    if st == JobState.FAILED.value:
        return JobResultResponse(
            job_id=job_id,
            status=st,
            error=status_data.get("error", "Job execution failed"),
        )

    if not metrics_file.exists():
        return JobResultResponse(
            job_id=job_id,
            status=st,
            elapsed_s=status_data.get("elapsed_s", 0.0),
        )

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        output_files = {
            "registered": str(job_dir / "output" / "registered.tif"),
            "residual_map": str(job_dir / "output" / "residual_map.png"),
            "metrics": str(metrics_file),
        }

        return JobResultResponse(
            job_id=job_id,
            status=st,
            rmse_px=metrics.get("rmse_px"),
            inlier_ratio=metrics.get("inlier_ratio"),
            sdi=metrics.get("sdi"),
            n_inliers=metrics.get("n_inliers"),
            n_total=metrics.get("n_total"),
            elapsed_s=metrics.get("elapsed_s"),
            transform=metrics.get("transform"),
            output_files=output_files,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {e}")


@app.get("/jobs/{job_id}/summary", response_model=ChatbotSummaryResponse)
def get_job_summary(job_id: str):
    """
    Unified Chatbot-Ready Output Endpoint (Schema v1.0).
    Assembles geodetic metrics, plain-language confidence classification,
    diagnostic reasoning, input metadata, and relative visual artifact paths.
    """
    try:
        summary = build_chatbot_summary(job_id)
        return summary
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as e:
        logger.error(f"Failed to build summary for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@app.get("/jobs/{job_id}/preview")
def get_job_preview(job_id: str, kind: str = "registered"):
    """
    Serve preview images for frontend visualization:
      - kind='registered'   : the warped registered GeoTIFF/image
      - kind='residual'     : the 2D residual error heatmap
      - kind='checkerboard' : alternating tile mosaic of source vs warped reference
      - kind='tiepoints'    : side-by-side keypoint correspondence overlay
    """
    job_dir = Path("data") / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check job completion for derived overlays
    status_file = job_dir / "status.json"
    job_status = "UNKNOWN"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                job_status = json.load(f).get("status", "UNKNOWN")
        except Exception:
            pass

    if kind == "residual":
        preview_path = job_dir / "output" / "residual_map.png"
        if preview_path.exists():
            return FileResponse(str(preview_path), media_type="image/png")
        raise HTTPException(status_code=404, detail="Residual map preview not yet generated.")

    elif kind == "checkerboard":
        if job_status != JobState.DONE.value:
            raise HTTPException(status_code=400, detail=f"Job is not completed (current status: {job_status})")

        cb_cache_path = job_dir / "output" / "preview_checkerboard.png"
        if cb_cache_path.exists():
            return FileResponse(str(cb_cache_path), media_type="image/png")

        path_a, _ = _find_job_inputs(job_dir)
        if not path_a or not os.path.exists(path_a):
            raise HTTPException(status_code=404, detail="Source image input path not recorded or found for checkerboard.")

        # Load raw_a
        try:
            raw_a, _ = read_raster(path_a)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read source image: {e}")

        # Load registered image
        registered_path = job_dir / "output" / "registered.tif"
        if registered_path.exists() and _HAS_RASTERIO:
            with rasterio.open(str(registered_path)) as s:
                registered_img = s.read(1).astype(np.float64)
        elif (job_dir / "output" / "registered.npy").exists():
            registered_img = np.load(str(job_dir / "output" / "registered.npy"))
        else:
            raise HTTPException(status_code=404, detail="Registered output raster not found.")

        checkerboard = create_checkerboard(raw_a, registered_img, tile_size=36)
        cb_u8 = (np.clip(checkerboard, 0, 1) * 255.0).astype(np.uint8)
        cv2.imwrite(str(cb_cache_path), cb_u8)
        return FileResponse(str(cb_cache_path), media_type="image/png")

    elif kind == "tiepoints":
        if job_status != JobState.DONE.value:
            raise HTTPException(status_code=400, detail=f"Job is not completed (current status: {job_status})")

        tp_cache_path = job_dir / "output" / "preview_tiepoints.png"
        if tp_cache_path.exists():
            return FileResponse(str(tp_cache_path), media_type="image/png")

        path_a, path_b = _find_job_inputs(job_dir)
        if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
            raise HTTPException(status_code=404, detail="Input images not recorded or found on disk.")

        matches_path = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_path.exists():
            matches_path = job_dir / "intermediate" / "matches_raw.npy"
        if not matches_path.exists():
            raise HTTPException(status_code=404, detail="Match correspondences not found.")

        try:
            raw_a, _ = read_raster(path_a)
            raw_b, _ = read_raster(path_b)
            matches = np.load(str(matches_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read data for tie-points overlay: {e}")

        tp_canvas = draw_tie_points(raw_a, raw_b, matches)
        cv2.imwrite(str(tp_cache_path), tp_canvas)
        return FileResponse(str(tp_cache_path), media_type="image/png")

    # Default: registered output
    tif_path = job_dir / "output" / "registered.tif"
    if tif_path.exists():
        return FileResponse(str(tif_path), media_type="image/tiff")

    npy_path = job_dir / "output" / "registered.npy"
    if npy_path.exists():
        return FileResponse(str(npy_path), media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail=f"Preview kind '{kind}' not found or output missing.")



# ── Chat & Interactive Follow-Up Endpoints ────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat_with_registration(req: ChatRequest):
    """
    Interactive conversational assistant grounded in registration telemetry.
    Uses Groq LLM (llama-3.3-70b-versatile) for low-latency reasoning.
    """
    job_id = req.job_id
    session_key = req.session_id or job_id
    message = req.message

    # a) Load the job's summary from disk
    summary_path = Path(f"data/jobs/{job_id}/output/summary.json")
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read summary file: {e}")
    else:
        try:
            summary = build_chatbot_summary(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")

    if summary.get("status") != "DONE":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} is not completed (current status: {summary.get('status')})"
        )

    # b) Build system prompt
    qa = summary.get("quality_assessment", {})
    metrics = summary.get("metrics", {})
    meta = summary.get("input_metadata", {})

    grade = qa.get("grade", "N/A")
    conf = qa.get("confidence_label", "N/A")
    rmse = metrics.get("rmse_px")
    rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
    inlier_ratio = metrics.get("inlier_ratio")
    inlier_ratio_str = f"{inlier_ratio * 100:.1f}" if inlier_ratio is not None else "N/A"
    n_inliers = metrics.get("n_inliers", 0)
    n_total = metrics.get("n_total", 0)
    sdi = metrics.get("sdi")
    sdi_str = f"{sdi:.4f}" if sdi is not None else "N/A"
    transform = metrics.get("transform_type", "homography")
    scale_disp = meta.get("scale_disparity_ratio", 1.0)
    solar_corr = meta.get("solar_correction_applied", False)
    warnings = qa.get("warnings", [])
    warnings_str = '; '.join(warnings) if warnings else 'none'
    reasoning = qa.get("reasoning", "")

    system = f"""You are LUNA-MATCH, a scientific assistant for lunar image registration. You have access to the results of a completed registration job. Answer the scientist's questions about these results accurately and concisely. Do not invent numbers — only reference the metrics below.

Registration summary:
- Grade: {grade} ({conf})
- RMSE: {rmse_str} px (sub-pixel = <0.5px, good)
- Inlier ratio: {inlier_ratio_str}% (robust = >30%, excellent = >80%)
- Inliers: {n_inliers} / {n_total} keypoint matches verified
- SDI: {sdi_str} (spatial coverage, good = >0.6)
- Transform: {transform}
- Scale disparity: {scale_disp:.2f}x between sensors
- Solar correction applied: {solar_corr}
- Warnings: {warnings_str}
- Scientific reasoning: {reasoning}

Keep answers under 150 words unless the scientist asks for more detail.
If asked about something outside these metrics, say you can only discuss this specific registration result."""

    # c) Build conversation history
    history = chat_sessions[session_key]
    messages = history + [{"role": "user", "content": message}]

    # d) Call Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chat not configured: GROQ_API_KEY missing"
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=300,
            temperature=0.3,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    # e) Save turn to history
    chat_sessions[session_key].append({"role": "user", "content": message})
    chat_sessions[session_key].append({"role": "assistant", "content": reply})

    # f) Return response
    turn = len(chat_sessions[session_key]) // 2
    return ChatResponse(
        session_id=session_key,
        reply=reply,
        turn=turn,
        job_id=job_id,
    )


@app.get("/chat/{job_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(job_id: str, session_id: Optional[str] = None):
    """
    Retrieve conversation history so frontend can restore chat on page reload.
    """
    session_key = session_id or job_id
    turns = chat_sessions.get(session_key, [])
    return ChatHistoryResponse(
        job_id=job_id,
        session_id=session_key,
        turns=turns,
    )


# ── Visual Analysis Endpoint ─────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_registration(job_id: str, focus: str = "overall"):
    """
    One-shot visual analysis of a completed registration job.
    Uses Groq LLM to provide scientific commentary on the registration quality.
    Focus options: 'overall', 'tiepoints', 'residual'.
    
    Note: This endpoint provides text-based analysis using registration metrics.
    Visual model support depends on Groq tier availability.
    """
    import time

    # a) Load summary
    summary_path = Path(f"data/jobs/{job_id}/output/summary.json")
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read summary: {e}")
    else:
        try:
            summary = build_chatbot_summary(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")

    if summary.get("status") != "DONE":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} is not completed (current status: {summary.get('status')})"
        )

    # b) Pick image based on focus
    focus_lower = focus.lower()
    if focus_lower == "residual":
        kind = "residual"
        image_path = Path(f"data/jobs/{job_id}/output/residual_map.png")
        analysis_context = "residual error heatmap showing spatial distribution of reprojection errors"
    elif focus_lower == "tiepoints":
        kind = "tiepoints"
        image_path = Path(f"data/jobs/{job_id}/output/preview_tiepoints.png")
        analysis_context = "tiepoints correspondence overlay showing keypoint match distribution"
    else:  # overall / checkerboard
        kind = "checkerboard"
        image_path = Path(f"data/jobs/{job_id}/output/preview_checkerboard.png")
        analysis_context = "checkerboard mosaic overlay comparing aligned images tile by tile"

    # Ensure image exists (regenerate via preview endpoint logic if needed)
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Preview image for focus='{focus}' not found. Call GET /jobs/{job_id}/preview?kind={kind} first to generate it."
        )

    # c) Extract metrics for prompt
    qa = summary.get("quality_assessment", {})
    metrics = summary.get("metrics", {})
    meta = summary.get("input_metadata", {})

    grade = qa.get("grade", "N/A")
    conf = qa.get("confidence_label", "N/A")
    rmse = metrics.get("rmse_px")
    rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
    inlier_ratio = metrics.get("inlier_ratio")
    inlier_ratio_str = f"{inlier_ratio * 100:.1f}" if inlier_ratio is not None else "N/A"
    n_inliers = metrics.get("n_inliers", 0)
    n_total = metrics.get("n_total", 0)
    sdi = metrics.get("sdi")
    sdi_str = f"{sdi:.4f}" if sdi is not None else "N/A"
    transform = metrics.get("transform_type", "homography")
    scale_disp = meta.get("scale_disparity_ratio", 1.0)
    solar_corr = meta.get("solar_correction_applied", False)
    warnings_list = qa.get("warnings", [])
    warnings_str = "; ".join(warnings_list) if warnings_list else "none"
    reasoning = qa.get("reasoning", "")

    # d) Build analysis prompt
    prompt = f"""You are a lunar image registration analyst. Provide a brief scientific assessment of this registration result (under 120 words).

You are analyzing the {analysis_context} for job: {job_id}

The registration metrics are:
- Grade: {grade} ({conf})
- RMSE: {rmse_str} px (sub-pixel threshold: <0.5px)
- Inlier ratio: {inlier_ratio_str}% ({n_inliers} verified out of {n_total} total keypoints)
- SDI: {sdi_str} (spatial coverage; good = >0.6)
- Transform: {transform}
- Scale disparity: {scale_disp:.2f}x between sensors
- Solar correction applied: {solar_corr}
- Warnings: {warnings_str}

Based on these metrics, provide your analysis:
1. Whether the keypoint spatial distribution (SDI={sdi_str}) looks well-spread
2. Whether the RMSE of {rmse_str}px indicates reliable sub-pixel alignment
3. Whether the inlier ratio of {inlier_ratio_str}% indicates robust matching
4. One-sentence conclusion on suitability for scientific lunar crater mapping.

Be factual, concise, and reference the specific numbers above."""

    # e) Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chat not configured: GROQ_API_KEY missing"
        )

    # f) Try vision call first, fall back to text analysis
    VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "")
    TEXT_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

    t0 = time.time()
    model_used = TEXT_MODEL

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # Try vision model with image if configured and image exists
        if VISION_MODEL and image_path.exists():
            try:
                import base64
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    max_tokens=250,
                    temperature=0.2,
                )
                visual_analysis = response.choices[0].message.content
                model_used = VISION_MODEL
            except Exception as vision_error:
                logger.warning(f"Vision model failed ({VISION_MODEL}): {vision_error}. Falling back to text analysis.")
                # Fall back to text-only analysis
                response = client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[{
                        "role": "system",
                        "content": "You are a lunar image registration analyst. Provide scientific metric-based analysis."
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=250,
                    temperature=0.2,
                )
                visual_analysis = response.choices[0].message.content
        else:
            # Text-only analysis using metrics
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{
                    "role": "system",
                    "content": "You are a lunar image registration analyst. Provide scientific metric-based analysis."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=250,
                temperature=0.2,
            )
            visual_analysis = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    latency = round(time.time() - t0, 3)

    # g) Return response
    return AnalyzeResponse(
        job_id=job_id,
        visual_analysis=visual_analysis,
        image_used=kind,
        model_used=model_used,
        latency_s=latency,
    )


def _decode_b64_image(b64_str: Optional[str]) -> Optional[np.ndarray]:
    """Decode a base64 data URI or raw base64 string into a grayscale OpenCV numpy array."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def _encode_b64_image(img: np.ndarray) -> str:
    """Encode an image numpy array into a PNG data URI string."""
    if img is None:
        return ""
    if img.dtype != np.uint8:
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        img_u8 = (np.clip(norm, 0, 1) * 255.0).astype(np.uint8)
    else:
        img_u8 = img
    success, buffer = cv2.imencode(".png", img_u8)
    if not success:
        return ""
    b64_bytes = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_bytes}"


@app.post("/orchestrate", response_model=OrchestrateResponse, tags=["orchestrate"])
def orchestrate(req: OrchestrateRequest):
    """
    Unified AI Orchestrator endpoint for frontend chat & analysis workspace.
    Supports:
    1. Image registration: accepts source_image_b64 and reference_image_b64
    2. Metric explanation & follow-up Q&A: accepts query + current_registration
    3. General lunar science knowledge Q&A
    """
    query = (req.query or "").strip()
    source_b64 = req.source_image_b64
    ref_b64 = req.reference_image_b64
    current_reg = req.current_registration or req.registration_result

    # ── CASE 1: Image Registration ──────────────────────────────────────────
    if source_b64 and ref_b64:
        img_a = _decode_b64_image(source_b64)
        img_b = _decode_b64_image(ref_b64)

        if img_a is None or img_b is None:
            raise HTTPException(status_code=400, detail="Could not decode one or both base64 images.")

        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job_dir = Path("data") / "jobs" / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        path_a = input_dir / "source.png"
        path_b = input_dir / "reference.png"
        cv2.imwrite(str(path_a), img_a)
        cv2.imwrite(str(path_b), img_b)

        pipeline = LunaMatchPipeline(job_id, str(path_a), str(path_b))
        pipeline_res = pipeline.run()

        summary = build_chatbot_summary(job_id)
        m = summary.get("metrics", {})
        qa = summary.get("quality_assessment", {})

        total_matches = int(m.get("n_total", 0))
        inliers = int(m.get("n_inliers", 0))
        inlier_ratio = float(m.get("inlier_ratio", 0.0))
        rmse = float(m.get("rmse_px", 0.0))
        subpixel_acc = rmse < 1.0

        # Build preview artifacts in base64
        registered_b64 = ""
        overlay_b64 = ""
        match_b64 = ""

        # Registered image
        reg_npy_path = job_dir / "output" / "registered.npy"
        reg_tif_path = job_dir / "output" / "registered.tif"
        reg_img = None
        if reg_npy_path.exists():
            reg_img = np.load(str(reg_npy_path))
        elif reg_tif_path.exists() and _HAS_RASTERIO:
            with rasterio.open(str(reg_tif_path)) as src:
                reg_img = src.read(1).astype(np.float64)

        if reg_img is not None:
            registered_b64 = _encode_b64_image(reg_img)
            # Checkerboard overlay
            raw_a, _ = read_raster(str(path_a))
            checker = create_checkerboard(raw_a, reg_img, tile_size=36)
            overlay_b64 = _encode_b64_image(checker)

        # Tie points matches
        matches_npy = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_npy.exists():
            matches_npy = job_dir / "intermediate" / "matches_raw.npy"
        if matches_npy.exists():
            try:
                matches_arr = np.load(str(matches_npy))
                raw_a, _ = read_raster(str(path_a))
                raw_b, _ = read_raster(str(path_b))
                tp_img = draw_tie_points(raw_a, raw_b, matches_arr)
                match_b64 = _encode_b64_image(tp_img)
            except Exception as e:
                logger.warning(f"Failed to generate tie-points visualization: {e}")

        # Text explanation
        groq_key = os.environ.get("GROQ_API_KEY")
        text_resp = qa.get("reasoning", "")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                prompt = (
                    f"You are the LUNA-MATCH AI assistant for ISRO Chandrayaan-2 lunar image registration.\n"
                    f"A registration job just finished with these metrics:\n"
                    f"- Grade: {qa.get('grade')}\n"
                    f"- Confidence: {qa.get('confidence_label')}\n"
                    f"- Inliers: {inliers}/{total_matches} ({inlier_ratio*100:.1f}%)\n"
                    f"- RMSE: {rmse:.3f} px (Sub-pixel: {subpixel_acc})\n"
                    f"Explain this result clearly to a planetary scientist in 2 concise paragraphs. "
                    f"Mention the alignment quality and whether it is safe for DEM/orthorectification."
                )
                groq_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
                completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "You are an expert planetary remote sensing specialist at ISRO."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                text_resp = completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq explanation failed: {e}")

        reg_result = {
            "status": "success" if pipeline_res.get("status") == JobState.DONE.value else "failed",
            "job_id": job_id,
            "total_matches": total_matches,
            "inliers": inliers,
            "outliers": max(0, total_matches - inliers),
            "inlier_ratio": inlier_ratio,
            "rmse": rmse,
            "subpixel_error": rmse,
            "subpixel_accuracy": subpixel_acc,
            "transformation_matrix": pipeline_res.get("transform"),
            "registered_image": registered_b64,
            "overlay_image": overlay_b64,
            "match_points_image": match_b64,
        }

        return OrchestrateResponse(
            intent="register_images",
            text_response=text_resp,
            registration_result=reg_result,
            sources=[],
            tools_called=["register_images"],
        )

    # ── CASE 2: Follow-up question on existing registration ─────────────────
    elif current_reg:
        groq_key = os.environ.get("GROQ_API_KEY")
        user_query = query or "Explain the registration metrics."
        text_resp = (
            f"The current registration achieved an RMSE of {current_reg.get('rmse', 'N/A')} "
            f"with {current_reg.get('inliers', 'N/A')} confirmed inliers. "
            f"Sub-pixel accuracy is {'verified' if current_reg.get('subpixel_accuracy') else 'unverified'}."
        )
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                context = (
                    f"Registration metrics:\n"
                    f"- Matches: {current_reg.get('total_matches')}\n"
                    f"- Inliers: {current_reg.get('inliers')}\n"
                    f"- Ratio: {current_reg.get('inlier_ratio')}\n"
                    f"- RMSE: {current_reg.get('rmse')} px\n"
                    f"- Sub-pixel accuracy: {current_reg.get('subpixel_accuracy')}\n"
                )
                groq_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
                completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "You are LUNA-MATCH AI assistant, an expert in lunar surface registration and photometry. Answer scientifically and directly."},
                        {"role": "user", "content": f"{context}\nQuestion: {user_query}"}
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                text_resp = completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq chat failed: {e}")

        return OrchestrateResponse(
            intent="explain_registration",
            text_response=text_resp,
            registration_result=current_reg,
            sources=[],
            tools_called=["explain_registration"],
        )

    # ── CASE 3: General lunar science chat ──────────────────────────────────
    else:
        user_query = query or "What is LUNA-MATCH?"
        groq_key = os.environ.get("GROQ_API_KEY")
        text_resp = (
            "LUNA-MATCH is an automated sub-pixel image registration and cross-modal alignment system "
            "designed for Chandrayaan-2 planetary imagery (OHRC, TMC-2, IIRS) and reference basemaps."
        )
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                groq_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
                completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "You are LUNA-MATCH AI assistant, an expert in Chandrayaan-2 lunar payloads (OHRC, TMC-2, IIRS) and planetary image correspondence."},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                text_resp = completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq chat failed: {e}")

        return OrchestrateResponse(
            intent="general_knowledge",
            text_response=text_resp,
            registration_result=None,
            sources=[],
            tools_called=[],
        )
