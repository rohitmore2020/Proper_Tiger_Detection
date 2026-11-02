from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import time
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.db import init_db, insert_detection, get_history, get_detection_by_id
from app.services.detector import get_detector
from app.services.reid_engine import get_reid_engine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Tiger Detection and Re-Identification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (Vite default is 5173, also permit local previews)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference images and uploaded images directories
REFERENCE_DIR = Path("dataset/reference")
if not REFERENCE_DIR.exists():
    REFERENCE_DIR = Path("../dataset/reference")

UPLOADS_DIR = Path("backend/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if REFERENCE_DIR.exists():
    app.mount("/static/reference", StaticFiles(directory=str(REFERENCE_DIR)), name="reference_images")

app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="upload_images")


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    detected: bool
    box: List[float]
    confidence: float


class Candidate(BaseModel):
    tiger_id: str
    name: str
    similarity_score: float
    reference_image: Optional[str] = None


class IdentificationResult(BaseModel):
    matched_id: str
    name: str
    similarity_score: float
    is_known: bool
    reference_image: Optional[str] = None
    top_candidates: List[Candidate] = []


class AnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    image_url: Optional[str] = None
    width: int
    height: int
    detection: DetectionResult
    identification: IdentificationResult
    processing_time_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Tiger Detection and Re-Identification API",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/history")
async def fetch_history(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    records = await get_history(limit=limit, offset=offset)
    return {
        "total": len(records),
        "limit": limit,
        "offset": offset,
        "records": records,
    }


@app.get("/api/v1/history/{record_id}")
async def fetch_detection_record(record_id: int):
    record = await get_detection_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Detection record not found.")
    return record


@app.post("/api/v1/images/validate")
async def validate_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()

        with Image.open(BytesIO(content)) as image:
            return {
                "filename": file.filename,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "size_bytes": len(content),
                "status": "valid",
            }
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")


@app.post("/api/v1/detect-and-identify", response_model=AnalysisResponse)
async def detect_and_identify(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a valid image file (JPEG, PNG, WebP).")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")

    # Save to uploads for static preview
    now_dt = datetime.now()
    timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{timestamp_str}_{file.filename}"
    save_path = UPLOADS_DIR / saved_filename
    image.save(save_path)
    image_url = f"/static/uploads/{saved_filename}"

    # 1. Tiger Detection & Crop
    detector = get_detector()
    detected, bbox, det_conf, crop = detector.detect_and_crop(image)

    # 2. Individual Re-Identification
    reid_engine = get_reid_engine()
    reid_res = reid_engine.identify(crop)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # 3. Persist to SQLite Database
    record_id = await insert_detection({
        "timestamp": now_dt.isoformat(),
        "filename": file.filename,
        "image_url": image_url,
        "width": image.width,
        "height": image.height,
        "detected": detected,
        "bbox_x1": bbox[0] if bbox else 0.0,
        "bbox_y1": bbox[1] if bbox else 0.0,
        "bbox_x2": bbox[2] if bbox else float(image.width),
        "bbox_y2": bbox[3] if bbox else float(image.height),
        "detection_confidence": det_conf,
        "predicted_id": reid_res["matched_id"],
        "tiger_name": reid_res["name"],
        "similarity_score": reid_res["similarity_score"],
        "is_known": reid_res["is_known"],
        "reference_image": reid_res["reference_image"],
        "processing_time_ms": elapsed_ms,
    })

    return AnalysisResponse(
        id=record_id,
        filename=saved_filename,
        image_url=image_url,
        width=image.width,
        height=image.height,
        detection=DetectionResult(
            detected=detected,
            box=bbox,
            confidence=det_conf,
        ),
        identification=IdentificationResult(
            matched_id=reid_res["matched_id"],
            name=reid_res["name"],
            similarity_score=reid_res["similarity_score"],
            is_known=reid_res["is_known"],
            reference_image=reid_res["reference_image"],
            top_candidates=[
                Candidate(
                    tiger_id=c["tiger_id"],
                    name=c["name"],
                    similarity_score=c["similarity_score"],
                    reference_image=c["reference_image"],
                )
                for c in reid_res.get("top_candidates", [])
            ],
        ),
        processing_time_ms=elapsed_ms,
    )
