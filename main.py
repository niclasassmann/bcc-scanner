"""
BCC Card Scanner — Slab Detection & Image Enhancement Service
Pure OpenCV pipeline, no AI/ML dependencies required for geometry.
"""

import cv2
import numpy as np
import io
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BCC Card Scanner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Quality-Score", "X-Preview-Image", "X-Rejection-Reason"],
)

# ── Constants ────────────────────────────────────────────────────────────────

OUTPUT_W, OUTPUT_H = 1200, 1680          # Final scan resolution (portrait)
PREVIEW_W, PREVIEW_H = 600, 840         # Preview resolution
SLAB_RATIO = OUTPUT_W / OUTPUT_H        # ≈ 0.714  (width / height)
RATIO_TOLERANCE = 0.40                  # Accept ratios within ±40% of target
MIN_AREA_FRACTION = 0.10                # Slab must cover at least 10% of image
MAX_AREA_FRACTION = 0.97                # Slab shouldn't be the entire image
MAX_LONG_EDGE = 1800                    # Resize input to this before processing
MIN_LAPLACIAN_VARIANCE = 60             # Below this → reject as too blurry
MIN_CONFIDENCE = 0.12                   # Below this → reject as no slab found


# ── Utilities ────────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image. Send a valid JPEG or PNG.")
    return img


def resize_for_processing(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale so longest edge = MAX_LONG_EDGE. Returns (resized, scale)."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= MAX_LONG_EDGE:
        return img.copy(), 1.0
    scale = MAX_LONG_EDGE / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # top-left      (smallest sum)
        pts[np.argmin(d)],   # top-right     (smallest diff)
        pts[np.argmax(s)],   # bottom-right  (largest sum)
        pts[np.argmax(d)],   # bottom-left   (largest diff)
    ], dtype=np.float32)


def angle_between(c: np.ndarray) -> float:
    """Mean interior angle deviation from 90° for a 4-point contour (degrees)."""
    deviations = []
    for i in range(4):
        p0 = c[(i - 1) % 4]
        p1 = c[i]
        p2 = c[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        deviations.append(abs(angle - 90))
    return float(np.mean(deviations))


# ── Detection ────────────────────────────────────────────────────────────────

def detect_slab(img: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Find the graded card slab in *img*.
    Returns (ordered 4-corner array, confidence) or (None, 0.0) if not found.
    """
    h, w = img.shape[:2]
    img_area = h * w
    cx, cy = w / 2, h / 2

    # 1. Grayscale + CLAHE for even contrast across lighting conditions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    best_corners = None
    best_score = 0.0

    # We try two preprocessing paths and keep whichever finds a better slab.
    # Path A: standard Canny.  Path B: aggressive blur + Canny (handles glare).
    for blur_k in (5, 11):
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, 25, 110)

        # Dilate edges slightly so thin slab borders close up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * MIN_AREA_FRACTION:
                continue
            if area > img_area * MAX_AREA_FRACTION:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            corners = approx.reshape(4, 2).astype(np.float32)

            # ── Score 1: aspect ratio ────────────────────────────────────────
            rect = cv2.minAreaRect(corners)
            rw, rh = sorted(rect[1])           # rw ≤ rh always
            if rh < 1:
                continue
            detected_ratio = rw / rh           # always < 1 (portrait)
            ratio_diff = abs(detected_ratio - SLAB_RATIO) / SLAB_RATIO
            if ratio_diff > RATIO_TOLERANCE:
                continue
            ratio_score = max(0.0, 1.0 - ratio_diff / RATIO_TOLERANCE)

            # ── Score 2: area coverage ───────────────────────────────────────
            frac = area / img_area
            area_score = 1.0 if MIN_AREA_FRACTION <= frac <= MAX_AREA_FRACTION else 0.0

            # ── Score 3: centrality ──────────────────────────────────────────
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            qcx = M["m10"] / M["m00"]
            qcy = M["m01"] / M["m00"]
            dist = np.hypot(qcx - cx, qcy - cy)
            max_dist = np.hypot(cx, cy)
            centrality_score = max(0.0, 1.0 - (dist / max_dist) / 0.7)

            # ── Score 4: rectangularity ──────────────────────────────────────
            mean_dev = angle_between(corners)
            rect_score = max(0.0, 1.0 - mean_dev / 25.0)

            # ── Score 5: area ratio to convex hull (solidity) ────────────────
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            solidity_score = solidity  # 1.0 for a perfect rectangle

            confidence = (
                ratio_score ** 1.5       # weight ratio heavily
                * area_score
                * centrality_score
                * rect_score
                * solidity_score
            )

            if confidence > best_score:
                best_score = confidence
                best_corners = order_corners(corners)

    return best_corners, best_score


# ── Sub-pixel refinement ─────────────────────────────────────────────────────

def refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), criteria)
    return refined


# ── Perspective warp ─────────────────────────────────────────────────────────

def warp_slab(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([
        [0, 0],
        [OUTPUT_W - 1, 0],
        [OUTPUT_W - 1, OUTPUT_H - 1],
        [0, OUTPUT_H - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (OUTPUT_W, OUTPUT_H),
                               flags=cv2.INTER_LANCZOS4)


# ── Post-processing ──────────────────────────────────────────────────────────

def check_blur(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def remove_glare(img: np.ndarray) -> tuple[np.ndarray, float]:
    """In-paint specular highlights. Returns (fixed image, glare fraction)."""
    mask = np.all(img > 245, axis=2).astype(np.uint8) * 255
    glare_frac = mask.sum() / 255 / mask.size
    if glare_frac > 0.001:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    return img, glare_frac


def flatten_illumination(img: np.ndarray) -> np.ndarray:
    """Divide by a very blurred version to remove lighting gradients."""
    h, w = img.shape[:2]
    k = max(int(min(h, w) * 0.25) | 1, 51)   # ensure odd, at least 51
    img_f = img.astype(np.float32)
    background = cv2.GaussianBlur(img_f, (k, k), 0)
    mean_lum = background.mean()
    corrected = np.clip((img_f / (background + 1e-6)) * mean_lum, 0, 255)
    return corrected.astype(np.uint8)


def white_balance(img: np.ndarray) -> np.ndarray:
    """
    Use the top 25% of the slab (label area, typically white/light)
    as a white reference and neutralise the colour cast.
    """
    h = img.shape[0]
    label_region = img[:h // 4, :]
    means = label_region.mean(axis=(0, 1))          # [B, G, R]
    if means.min() < 10:
        return img                                   # avoid divide-by-zero
    target = 210.0
    scales = target / means
    scales = np.clip(scales, 0.5, 2.5)              # guard extreme corrections
    balanced = np.clip(img.astype(np.float32) * scales, 0, 255)
    return balanced.astype(np.uint8)


def enhance_colors_and_sharpness(img: np.ndarray) -> np.ndarray:
    # Saturation boost in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Unsharp mask for sharpening
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # CLAHE on L channel in LAB for local contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img


def correct_orientation(img: np.ndarray) -> np.ndarray:
    """
    Label area is lighter than card area.
    If the bottom third is brighter than the top third, rotate 180°.
    """
    h = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top_mean = float(gray[:h // 3, :].mean())
    bot_mean = float(gray[2 * h // 3:, :].mean())
    if bot_mean > top_mean + 12:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def barrel_undistort(img: np.ndarray, k1: float = -0.08) -> np.ndarray:
    """Apply a mild barrel distortion correction (generic phone camera model)."""
    h, w = img.shape[:2]
    fx = fy = max(w, h) * 1.0
    cx_, cy_ = w / 2, h / 2
    K = np.array([[fx, 0, cx_], [0, fy, cy_], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0, 0, 0], dtype=np.float64)
    return cv2.undistort(img, K, dist)


# ── Quality score ─────────────────────────────────────────────────────────────

def compute_quality(confidence: float, lap_var: float, glare_frac: float) -> float:
    blur_score = min(lap_var, 600) / 600
    glare_score = max(0.0, 1.0 - glare_frac * 20)
    return round((confidence + blur_score + glare_score) / 3, 3)


# ── Encode helpers ────────────────────────────────────────────────────────────

def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def to_jpeg_bytes(img: np.ndarray, quality: int = 88) -> bytes:
    preview = cv2.resize(img, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "BCC Card Scanner"}


@app.post("/crop-slab")
async def crop_slab(file: UploadFile = File(...)):
    raw = await file.read()
    original = decode_image(raw)

    # ── Step 1: resize for fast processing ───────────────────────────────────
    small, scale = resize_for_processing(original)

    # ── Step 2: detect slab ───────────────────────────────────────────────────
    corners, confidence = detect_slab(small)

    if corners is None or confidence < MIN_CONFIDENCE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_slab_detected",
                "message": "Could not find a graded card slab in this image. "
                           "Ensure the slab is visible and reasonably centred.",
                "confidence": round(confidence, 3),
            },
        )

    # ── Step 3: scale corners back to original resolution ─────────────────────
    if scale < 1.0:
        corners = corners / scale

    # ── Step 4: sub-pixel refinement on full-res grayscale ───────────────────
    gray_full = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    corners = refine_corners(gray_full, corners)

    # ── Step 5: perspective warp ──────────────────────────────────────────────
    warped = warp_slab(original, corners)

    # ── Step 6: blur check ────────────────────────────────────────────────────
    lap_var = check_blur(warped)
    if lap_var < MIN_LAPLACIAN_VARIANCE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "image_too_blurry",
                "message": "Hold the camera steady and ensure the slab is in focus.",
                "laplacian_variance": round(lap_var, 1),
            },
        )

    # ── Step 7: barrel distortion correction ─────────────────────────────────
    warped = barrel_undistort(warped)

    # ── Step 8: glare removal ─────────────────────────────────────────────────
    warped, glare_frac = remove_glare(warped)

    # ── Step 9: illumination flattening ──────────────────────────────────────
    warped = flatten_illumination(warped)

    # ── Step 10: white balance ────────────────────────────────────────────────
    warped = white_balance(warped)

    # ── Step 11: colour & sharpness enhancement ───────────────────────────────
    warped = enhance_colors_and_sharpness(warped)

    # ── Step 12: orientation correction ──────────────────────────────────────
    warped = correct_orientation(warped)

    # ── Step 13: encode outputs ───────────────────────────────────────────────
    png_bytes = to_png_bytes(warped)
    jpeg_bytes = to_jpeg_bytes(warped)
    preview_b64 = base64.b64encode(jpeg_bytes).decode()

    quality_score = compute_quality(confidence, lap_var, glare_frac)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Quality-Score": str(quality_score),
            "X-Preview-Image": preview_b64,
            "X-Confidence": str(round(confidence, 3)),
        },
    )
