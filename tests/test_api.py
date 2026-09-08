"""
tests/test_api.py
=================
Integration tests for the LUNA-MATCH FastAPI application.
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_images():
    td = tempfile.mkdtemp()
    img_a = np.random.uniform(0.1, 0.9, (64, 64))
    img_b = img_a + np.random.normal(0, 0.05, (64, 64))

    p_a = os.path.join(td, "img_a.npy")
    p_b = os.path.join(td, "img_b.npy")
    np.save(p_a, img_a)
    np.save(p_b, img_b)
    return p_a, p_b


def test_api_register_endpoint(client, synthetic_images):
    """POST /register initiates a job and returns HTTP 202."""
    p_a, p_b = synthetic_images
    resp = client.post("/register", json={
        "img_a_path": p_a,
        "img_b_path": p_b,
        "job_id": "test_api_job_001"
    })
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "test_api_job_001"
    assert data["status"] in ("PENDING", "PREPROCESSING", "DONE")


def test_api_get_job_status(client, synthetic_images):
    """GET /jobs/{id} returns job lifecycle state."""
    p_a, p_b = synthetic_images
    job_id = "test_api_status_002"
    client.post("/register", json={
        "img_a_path": p_a,
        "img_b_path": p_b,
        "job_id": job_id
    })
    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert "status" in data


def test_api_get_job_result_nonexistent(client):
    """GET /jobs/{id}/result returns 404 for nonexistent job."""
    resp = client.get("/jobs/nonexistent_id_999/result")
    assert resp.status_code == 404


from pathlib import Path
import json
from unittest.mock import patch, MagicMock


@pytest.fixture
def completed_job():
    job_id = "test_completed_job_chat_001"
    job_dir = Path("data") / "jobs" / job_id
    out_dir = job_dir / "output"
    in_dir = job_dir / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)

    with open(job_dir / "status.json", "w") as f:
        json.dump({"job_id": job_id, "status": "DONE", "elapsed_s": 1.25}, f)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "rmse_px": 0.3284,
            "inlier_ratio": 0.8924,
            "sdi": 0.8901,
            "transform": "homography",
            "n_inliers": 514,
            "n_total": 576,
            "elapsed_s": 1.25,
        }, f)

    with open(in_dir / "metadata.json", "w") as f:
        json.dump({
            "img_a_path": "moving.tif",
            "img_b_path": "ref.tif",
            "image_a": {"gsd": "100.0"},
            "image_b": {"gsd": "150.0"},
        }, f)

    yield job_id

    import shutil
    shutil.rmtree(job_dir, ignore_errors=True)


def test_chat_completed_job_success(client, completed_job):
    """POST /chat with completed job returns 200 with correct response shape."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Registration accuracy is Grade A with sub-pixel RMSE 0.3284 px."))]

    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}),          patch("groq.Groq") as mock_groq:
        mock_groq.return_value.chat.completions.create.return_value = mock_resp

        resp = client.post("/chat", json={
            "job_id": completed_job,
            "message": "Is this result reliable for crater mapping?",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == completed_job
        assert data["session_id"] == completed_job
        assert data["turn"] >= 1
        assert "Grade A" in data["reply"]


def test_chat_missing_job_returns_404(client):
    """POST /chat with missing job_id returns 404."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}):
        resp = client.post("/chat", json={
            "job_id": "nonexistent_job_xyz999",
            "message": "What is the error?",
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


def test_chat_history_grows_after_turns(client, completed_job):
    """Conversation history grows correctly after turns and is queryable via GET /chat/{id}/history."""
    mock_resp1 = MagicMock()
    mock_resp1.choices = [MagicMock(message=MagicMock(content="First reply."))]
    mock_resp2 = MagicMock()
    mock_resp2.choices = [MagicMock(message=MagicMock(content="Second reply."))]

    session_id = f"session_{completed_job}"

    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}),          patch("groq.Groq") as mock_groq:
        mock_groq.return_value.chat.completions.create.side_effect = [mock_resp1, mock_resp2]

        # Turn 1
        resp1 = client.post("/chat", json={
            "job_id": completed_job,
            "session_id": session_id,
            "message": "Turn 1 question",
        })
        assert resp1.status_code == 200
        assert resp1.json()["turn"] == 1

        # Turn 2
        resp2 = client.post("/chat", json={
            "job_id": completed_job,
            "session_id": session_id,
            "message": "Turn 2 question",
        })
        assert resp2.status_code == 200
        assert resp2.json()["turn"] == 2

        # Query history
        hist_resp = client.get(f"/chat/{completed_job}/history?session_id={session_id}")
        assert hist_resp.status_code == 200
        hist = hist_resp.json()
        assert len(hist["turns"]) == 4  # 2 user + 2 assistant messages
        assert hist["turns"][0]["content"] == "Turn 1 question"
        assert hist["turns"][1]["content"] == "First reply."
        assert hist["turns"][2]["content"] == "Turn 2 question"
        assert hist["turns"][3]["content"] == "Second reply."


def test_chat_missing_groq_api_key_returns_503(client, completed_job):
    """POST /chat returns 503 when GROQ_API_KEY environment variable is not set."""
    with patch.dict(os.environ, {}, clear=True):
        resp = client.post("/chat", json={
            "job_id": completed_job,
            "message": "Any question",
        })
        assert resp.status_code == 503
        assert "GROQ_API_KEY missing" in resp.json()["detail"]


def test_analyze_completed_job_success(client, completed_job):
    """POST /analyze with completed job returns 200 with correct shape."""
    # Create the required preview image
    import numpy as np
    from PIL import Image as PILImage
    out_dir = Path('data') / 'jobs' / completed_job / 'output'
    img = PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img.save(out_dir / 'preview_tiepoints.png')

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(
        content='Grade A with 89.2% inlier ratio indicates excellent alignment. SDI 0.8901 confirms well-distributed keypoints. Reliable for crater mapping.'
    ))]

    with patch.dict(os.environ, {'GROQ_API_KEY': 'mock_key'}),          patch('groq.Groq') as mock_groq:
        mock_groq.return_value.chat.completions.create.return_value = mock_resp
        resp = client.post('/analyze', params={'job_id': completed_job, 'focus': 'tiepoints'})

        assert resp.status_code == 200
        data = resp.json()
        assert data['job_id'] == completed_job
        assert 'visual_analysis' in data
        assert 'Grade A' in data['visual_analysis']
        assert data['image_used'] == 'tiepoints'
        assert isinstance(data['latency_s'], float)


def test_analyze_missing_job_returns_404(client):
    """POST /analyze with missing job_id returns 404."""
    with patch.dict(os.environ, {'GROQ_API_KEY': 'mock_key'}):
        resp = client.post('/analyze', params={'job_id': 'nonexistent_xyz_analyze_001', 'focus': 'overall'})
        assert resp.status_code == 404
        assert 'not found' in resp.json()['detail'].lower()


def test_analyze_missing_groq_key_returns_503(client, completed_job):
    """POST /analyze returns 503 when GROQ_API_KEY is not set."""
    import numpy as np
    from PIL import Image as PILImage
    out_dir = Path('data') / 'jobs' / completed_job / 'output'
    img = PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img.save(out_dir / 'preview_checkerboard.png')

    with patch.dict(os.environ, {}, clear=True):
        resp = client.post('/analyze', params={'job_id': completed_job, 'focus': 'overall'})
        assert resp.status_code == 503
        assert 'GROQ_API_KEY missing' in resp.json()['detail']
