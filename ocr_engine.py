"""
ocr_engine.py
-------------
Image loading, OpenCV preprocessing and OCR (EasyOCR).

Pipeline:  image -> resize -> grayscale -> denoise -> contrast (CLAHE)
           -> OCR  (and, if too little text was read, a second pass on a
                    thresholded image)
           -> words grouped into reading lines
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # optional: lets iPhone .HEIC photos be opened directly
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MAX_SIDE = 1800      # large photos are scaled down (faster OCR, less noise)
MIN_SIDE = 1100      # small photos are upscaled (small text becomes readable)
MIN_WORDS = 12       # fewer words than this triggers the thresholded second pass
GAP_FACTOR = 4.0     # horizontal gap (in text heights) that splits a row into columns



@dataclass
class OCRResult:
    lines: list[str]
    text: str
    mean_confidence: float
    word_count: int
    largest_text: str | None
    stages: dict = field(default_factory=dict)   # name -> image, for display
    pass_used: str = "enhanced"


def load_image(data: bytes) -> np.ndarray:
    """Bytes of any common image format -> upright BGR numpy array."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")   # respect phone rotation
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess(bgr: np.ndarray) -> dict:
    """Return every intermediate image so the steps can be shown in the UI."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
    elif longest < MIN_SIDE:
        scale = MIN_SIDE / longest
    else:
        scale = 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interp) if scale != 1 else bgr

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    thresholded = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )
    return {
        "Resized": resized,
        "Grayscale": gray,
        "Denoised": denoised,
        "Contrast enhanced": enhanced,
        "Thresholded": thresholded,
    }


@lru_cache(maxsize=1)
def get_reader():
    """Load the EasyOCR model once (first call downloads ~100 MB of weights)."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _group_into_lines(detections: list) -> tuple[list[str], str | None]:
    """EasyOCR returns separate word boxes; join boxes on the same row into a line."""
    boxes = []
    for bbox, text, conf in detections:
        if conf < 0.05 or not text.strip():
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        boxes.append({"text": text.strip(), "x": min(xs), "x2": max(xs), "yc": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys)})
    if not boxes:
        return [], None

    median_h = float(np.median([b["h"] for b in boxes]))
    boxes.sort(key=lambda b: b["yc"])
    rows: list[list[dict]] = []
    for b in boxes:
        if rows and abs(b["yc"] - np.mean([r["yc"] for r in rows[-1]])) < 0.55 * median_h:
            rows[-1].append(b)
        else:
            rows.append([b])

    # Within a row, a wide horizontal gap means a different text column
    # (e.g. printed dates beside an address), so it starts a new line.
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda b: b["x"])
        segment = [row[0]]
        for prev, b in zip(row, row[1:]):
            if b["x"] - prev["x2"] > GAP_FACTOR * median_h:
                lines.append(" ".join(x["text"] for x in segment))
                segment = []
            segment.append(b)
        lines.append(" ".join(x["text"] for x in segment))

    # The tallest readable word is usually the brand / product name.
    wordy = [b for b in boxes if sum(c.isalpha() for c in b["text"]) >= 3]
    largest = max(wordy, key=lambda b: b["h"])["text"] if wordy else None
    return lines, largest


def _read(image: np.ndarray) -> list:
    return get_reader().readtext(image, paragraph=False)


def run_ocr(bgr: np.ndarray) -> OCRResult:
    stages = preprocess(bgr)
    detections = _read(stages["Contrast enhanced"])
    pass_used = "enhanced"

    if sum(len(t.split()) for _, t, _ in detections) < MIN_WORDS:
        second = _read(stages["Thresholded"])
        if len(second) > len(detections):
            detections, pass_used = second, "thresholded"

    lines, largest = _group_into_lines(detections)
    confs = [c for _, _, c in detections] or [0.0]
    text = "\n".join(lines)
    return OCRResult(
        lines=lines,
        text=text,
        mean_confidence=float(np.mean(confs)),
        word_count=len(text.split()),
        largest_text=largest,
        stages=stages,
        pass_used=pass_used,
    )
