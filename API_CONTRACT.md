# LUNA-MATCH Core Engine API Contract

This document describes the interface specifications for the LUNA-MATCH Lunar Image Registration API. All example outputs below reflect real values verified during system execution.

---

## 1. GET /health

Service health status and diagnostic availability of matching components.

### Request
```http
GET /health HTTP/1.1
Host: localhost:8000
```

### Response (Real Output)
- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "status": "ok",
  "service": "LUNA-MATCH Registration API",
  "version": "2.0.0",
  "default_matcher": "classical",
  "structural_matching_available": true,
  "active_workers": 4
}
```

---

## 2. POST /register

Initiate an asynchronous registration job. Accepts multipart form file uploads (recommended for web clients/microservices) or JSON filepath specifications.

### Request (Multipart Form-Data)
```http
POST /register HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryX

------WebKitFormBoundaryX
Content-Disposition: form-data; name="img_a"; filename="verified_a.tif"
Content-Type: image/tiff

<binary data>
------WebKitFormBoundaryX
Content-Disposition: form-data; name="img_b"; filename="verified_b.tif"
Content-Type: image/tiff

<binary data>
------WebKitFormBoundaryX--
```

#### cURL Example:
```bash
curl -s -X POST http://localhost:8000/register   -F "img_a=@data/samples/verified_a.tif"   -F "img_b=@data/samples/verified_b.tif"
```

### Alternate Request (JSON Payload)
```http
POST /register HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "img_a_path": "data/samples/verified_a.tif",
  "img_b_path": "data/samples/verified_b.tif",
  "job_id": "custom_job_id_optional"
}
```

### Response (Real Output)
- **Status**: `202 Accepted`
- **Content-Type**: `application/json`

```json
{
  "job_id": "job_949cee5a6d",
  "status": "PENDING",
  "elapsed_s": 0.0,
  "error": null
}
```

---

## 3. GET /jobs/{id}

Query the current lifecycle state and execution progress of a registration job.

### Request
```http
GET /jobs/job_949cee5a6d HTTP/1.1
Host: localhost:8000
```

### Job Lifecycle Status Values
| Status | Meaning |
| :--- | :--- |
| `PENDING` | Job queued; workspace directories initialized. |
| `PREPROCESSING` | Reading rasters, GSD alignment, Lommel-Seeliger illumination normalization. |
| `MATCHING` | Extracting Phase Congruency / MIM structural descriptors and dense feature matching. |
| `VERIFYING` | Adaptive Non-Maximal Suppression (ANMS) and MAGSAC++ robust geometric filtering. |
| `REFINING` | Lucas-Kanade gradient-based sub-pixel tie-point refinement. |
| `DONE` | Image warped, RMSE/SDI metrics verified, and visual artifacts generated. |
| `FAILED` | An unrecoverable exception occurred (error message populated in `error` field). |

### Response (Real Output)
- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "job_id": "job_949cee5a6d",
  "status": "DONE",
  "elapsed_s": 1.21,
  "error": null
}
```

---

## 4. GET /jobs/{id}/summary

Unified Chatbot / VLM / RAG integration endpoint (Schema v1.1). Assembles geodetic metrics, plain-language confidence classification, diagnostic reasoning, input metadata, and relative visual artifact paths.

### Request
```http
GET /jobs/job_949cee5a6d/summary HTTP/1.1
Host: localhost:8000
```

### Response (Real Output Captured from Verification Run)
- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "schema_version": "1.1",
  "job_id": "job_949cee5a6d",
  "status": "DONE",
  "quality_assessment": {
    "confidence_label": "high confidence",
    "grade": "A",
    "reasoning": "Sub-pixel reprojection accuracy (RMSE 0.3284 px) with strong inlier verification (514 inliers, 89.2%) and solid spatial distribution (SDI 0.8901) under a verified homography transformation.",
    "warnings": [
      "Solar ephemeris angles missing in metadata; Lommel-Seeliger illumination normalization was bypassed."
    ]
  },
  "metrics": {
    "rmse_px": 0.3284,
    "inlier_ratio": 0.8924,
    "sdi": 0.8901,
    "transform_type": "homography",
    "n_inliers": 514,
    "n_total": 576,
    "elapsed_s": 1.21
  },
  "input_metadata": {
    "img_a_path": "data/jobs/job_949cee5a6d/input/upload_a_verified_a.tif",
    "img_b_path": "data/jobs/job_949cee5a6d/input/upload_b_verified_b.tif",
    "gsd_a_m_per_px": 131.6,
    "gsd_b_m_per_px": 219.3333,
    "scale_disparity_ratio": 1.67,
    "pixel_dimension_ratio": 1.0,
    "solar_correction_applied": false
  },
  "artifacts": {
    "registered_geotiff": "data/jobs/job_949cee5a6d/output/registered.tif",
    "checkerboard_png": "data/jobs/job_949cee5a6d/output/preview_checkerboard.png",
    "tiepoints_png": "data/jobs/job_949cee5a6d/output/preview_tiepoints.png",
    "residual_map_png": "data/jobs/job_949cee5a6d/output/residual_map.png"
  }
}
```

---

## 5. GET /jobs/{id}/preview

Serve binary image streams of generated artifacts and visualization overlays for web frontends, GIS inspection, or multimodal VLM ingestion.

### Request Parameters
`GET /jobs/{id}/preview?kind={kind}`

| Parameter | Type | Allowed Values | Content-Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| `kind=checkerboard` | Query | `checkerboard` | `image/png` | Alternating 8x8 checkerboard mosaic tile grid comparing source vs warped reference to visually detect edge discontinuities. |
| `kind=tiepoints` | Query | `tiepoints` | `image/png` | Side-by-side keypoint correspondence map showing verified inlier matches connected by colored vector lines. |
| `kind=residual` | Query | `residual` | `image/png` | 2D spatial heatmap of reprojection residuals across the overlapping image domain. |
| `kind=registered` | Query | `registered` | `image/tiff` | The georeferenced output raster warped into alignment with the reference frame. |

### cURL Example:
```bash
curl -s "http://localhost:8000/jobs/job_949cee5a6d/preview?kind=checkerboard" -o checkerboard.png
```
