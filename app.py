import os
import json
import glob
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from pathlib import Path
import tempfile, shutil
from typing import Optional, Dict, Any

# TensorFlow for Keras/SavedModel and TFLite
from tensorflow.keras.models import load_model as keras_load_model
import tensorflow as tf

st.set_page_config(page_title="🔬 NucleoNet", layout="wide")

# ============ Optional deps for advanced post-processing ============
try:
    from skimage.morphology import reconstruction, erosion, square, skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# ===============================
# Helpers: safe image display & normalization
# ===============================

def normalize_to_u8(arr):
    if isinstance(arr, Image.Image):
        arr = np.array(arr)
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - amin) / (amax - amin)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a

def convert_to_pil(img) -> Image.Image:
    if img is None:
        raise ValueError("convert_to_pil: obraz jest None")
    if isinstance(img, Image.Image):
        return img
    arr = normalize_to_u8(img)
    if not isinstance(arr, np.ndarray):
        raise TypeError("convert_to_pil: nieobsługiwany typ danych")
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3:
        # Jeśli źródło jest z OpenCV i ma BGR, przekształć je wcześniej do RGB!
        # Tu zakładamy RGB/RGBA.
        return Image.fromarray(arr)
    raise ValueError(f"convert_to_pil: nieoczekiwany kształt tablicy: {arr.shape}")

def show_u8(img, *args, **kwargs):
    """Display image safely (uint8), auto-normalizing if needed."""
    st.image(normalize_to_u8(img), *args, **kwargs)

# ===============================
# General helpers
# ===============================

def to_gray(img):
    """Convert to grayscale safely for 2, 3, or 4 channel images."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        img = normalize_to_u8(img)
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def crop_image(img, top, bottom, left, right):
    h, w = img.shape[:2]
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - 1))
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w - 1))
    if top + bottom >= h or left + right >= w:
        return img
    return img[top:h - bottom, left:w - right]

def apply_clahe(img, clip, grid):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(to_gray(img))

def apply_bilateral(img, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(to_gray(img), d, sigmaColor, sigmaSpace)

def apply_unsharp(img, ksize, sigma):
    g = to_gray(img)
    blur = cv2.GaussianBlur(g, (ksize, ksize), sigma)
    return cv2.addWeighted(g, 1.5, blur, -0.5, 0)

def apply_nlm_denoise(img, h_val):
    return cv2.fastNlMeansDenoising(to_gray(img), None, h_val, 7, 21)

def apply_threshold(img, block, C):
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    return cv2.adaptiveThreshold(
        to_gray(img), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block, C
    )

def show_histogram(image, title):
    arr = normalize_to_u8(image)
    hist, _ = np.histogram(arr.flatten(), 256, [0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_title(title)
    st.pyplot(fig)

# --- Thresholding helpers ---

def otsu_threshold(img, invert=False):
    g = to_gray(img)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = 255 - th
    return th

def adaptive_local_threshold(img, block=15, C=2, method="mean", invert=False):
    g = to_gray(img)
    block = int(block)
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    meth = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    th = cv2.adaptiveThreshold(g, 255, meth, cv2.THRESH_BINARY, block, int(C))
    if invert:
        th = 255 - th
    return th

def fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes in a binary 0/255 mask using floodFill."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    ff = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(m, holes)

def count_cells_filtered(mask_u8: np.ndarray, min_area_px: int, exclude_border: bool = True, do_fill_holes: bool = True):
    if mask_u8 is None:
        return 0, None
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if do_fill_holes:
        m = fill_holes_u8(m)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return 0, np.zeros_like(m)
    areas  = stats[1:, cv2.CC_STAT_AREA]
    left   = stats[1:, cv2.CC_STAT_LEFT]
    top    = stats[1:, cv2.CC_STAT_TOP]
    width  = stats[1:, cv2.CC_STAT_WIDTH]
    height = stats[1:, cv2.CC_STAT_HEIGHT]
    H, W = m.shape[:2]
    keep = areas >= int(max(1, min_area_px))
    if exclude_border:
        touch = (left == 0) | (top == 0) | (left + width >= W) | (top + height >= H)
        keep &= ~touch
    kept_mask = np.zeros_like(m, dtype=np.uint8)
    kept_labels = np.where(keep)[0] + 1
    for lab in kept_labels:
        kept_mask[labels == lab] = 255
    return int(keep.sum()), kept_mask

def morph_separate(mask_u8, op="none", ksize=3, iters=1):
    if op == "none":
        return mask_u8
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == "erode":
        return cv2.erode(mask_u8, kernel, iterations=int(iters))
    if op == "open":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=int(iters))
    if op == "close":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=int(iters))
    if op == "dilate":
        return cv2.dilate(mask_u8, kernel, iterations=int(iters))
    return mask_u8

def watershed_on_mask(mask_u8, min_peak_frac=0.6):
    """Split touching blobs using distance transform-based watershed on a binary mask."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, float(min_peak_frac) * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(m, sure_fg)
    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(color, markers)
    out = (markers_ws > 1).astype(np.uint8) * 255
    return out

def occupancy_from_mask(mask_u8):
    """Coverage exclusively from a 0/255 binary mask."""
    if mask_u8 is None:
        return 0, 0, 0.0
    g = (normalize_to_u8(mask_u8) > 0).astype(np.uint8)
    total = int(g.size)
    occupied = int(g.sum())
    percent = (occupied / total) * 100.0 if total > 0 else 0.0
    return occupied, total, percent

# ======== Advanced post-processing (requires scikit-image) ========

def opening_by_reconstruction(mask_u8: np.ndarray, size: int) -> np.ndarray:
    """Opening by reconstruction (binary). Erode then reconstruct within original mask."""
    if not SKIMAGE_AVAILABLE:
        st.warning("Opening by reconstruction requires scikit-image. Skipping.")
        return mask_u8
    size = max(1, int(size))
    m = (normalize_to_u8(mask_u8) > 0)
    seed = erosion(m, square(size))
    rec = reconstruction(seed.astype(np.uint8), m.astype(np.uint8), method='dilation')
    rec = (rec > 0).astype(np.uint8) * 255
    return rec

def skeletonize_binary(mask_u8: np.ndarray) -> np.ndarray:
    """One-pixel-wide skeleton of binary mask."""
    if not SKIMAGE_AVAILABLE:
        st.warning("Skeletonization requires scikit-image. Skipping.")
        return mask_u8
    m = (normalize_to_u8(mask_u8) > 0)
    sk = skeletonize(m)
    return (sk.astype(np.uint8)) * 255

# ===============================
# Configurable Watershed (on grayscale input)
# ===============================

def _odd(v):
    v = int(v)
    return v if v % 2 == 1 else v + 1

def watershed_segment(
    img,
    blur_type="median",
    blur_ksize=5,
    blur_sigma=1.0,
    thresh_mode="otsu",
    adaptive_block=11,
    adaptive_C=2,
    invert_binary=False,
    morph_kernel=3,
    opening_iter=2,
    dilate_iter=3,
    dist_type="L2",
    dist_mask=5,
    sure_fg_frac=0.7,
    connectivity=8,
    return_debug=False
):
    gray = to_gray(img)
    if blur_type == "median" and blur_ksize > 1:
        k = _odd(blur_ksize); work = cv2.medianBlur(gray, k)
    elif blur_type == "gaussian" and blur_ksize > 1:
        k = _odd(blur_ksize); work = cv2.GaussianBlur(gray, (k, k), float(blur_sigma))
    else:
        work = gray.copy()
    if thresh_mode == "otsu":
        _, thresh = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = _odd(adaptive_block)
        method = cv2.ADAPTIVE_THRESH_MEAN_C if thresh_mode == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresh = cv2.adaptiveThreshold(work, 255, method, cv2.THRESH_BINARY, block, int(adaptive_C))
    if invert_binary:
        thresh = 255 - thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(morph_kernel), _odd(morph_kernel)))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=int(opening_iter))
    sure_bg = cv2.dilate(opening, kernel, iterations=int(dilate_iter))
    dist_flag = cv2.DIST_L2 if dist_type == "L2" else (cv2.DIST_L1 if dist_type == "L1" else cv2.DIST_C)
    dist_transform = cv2.distanceTransform(opening, dist_flag, int(dist_mask))
    _, sure_fg = cv2.threshold(dist_transform, float(sure_fg_frac) * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cnt_cc, markers = cv2.connectedComponents(sure_fg, connectivity=int(connectivity))
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(color, markers)
    color[markers_ws == -1] = [255, 0, 0]
    ws_mask = (markers_ws > 1).astype(np.uint8) * 255
    if return_debug:
        dbg = {
            "gray": gray, "work_blur": work, "thresh": thresh, "opening": opening,
            "sure_bg": sure_bg,
            "dist_transform": dist_transform / (dist_transform.max() + 1e-8),
            "sure_fg": sure_fg, "unknown": unknown,
            "markers_initial": markers, "markers_ws": markers_ws.astype(np.int32),
            "components_count": cnt_cc
        }
        return color, ws_mask, dbg
    return color, ws_mask

# ===============================
# U-Net loaders (Keras/SavedModel/TFLite) + metadata
# ===============================

@st.cache_resource
def load_metadata() -> Dict[str, Any]:
    meta_paths = ["metadata.json", os.path.join("/mnt/data", "metadata.json")]
    for p in meta_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

META = load_metadata()
DEFAULT_W = int(META.get("img_width", 256))
DEFAULT_H = int(META.get("img_height", 256))
DEFAULT_THRESH = float(META.get("threshold", 0.5))
INPUT_NORM = META.get("normalization", "gray/255.0")
DEFAULT_MODEL_CANDIDATES = [
    META.get("default_model_path", ""),               # prefer metadata if provided
    "/mnt/data/model.tflite",
    "/mnt/data/model.keras",
    "model.tflite",
    "model.keras",
    "./saved_model",  # directory with saved_model.pb
]

class InferenceHandle:
    def __init__(self, kind: str, obj: Any, path: str):
        self.kind = kind
        self.obj = obj
        self.path = path

@st.cache_resource(show_spinner=False)
def load_unet_any(path: str) -> Optional[InferenceHandle]:
    if not path:
        return None
    path = os.path.abspath(path)
    if not (os.path.isfile(path) or os.path.isdir(path)):
        return None
    try:
        if path.endswith((".keras", ".h5")):
            model = keras_load_model(path, compile=False)
            return InferenceHandle("keras", model, path)
        if path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return InferenceHandle("tflite", interpreter, path)
        # SavedModel directory
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
            model = tf.keras.models.load_model(path, compile=False)
            return InferenceHandle("keras", model, path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return None

def segment_with_unet_handle(handle: InferenceHandle, image: np.ndarray, target_size=(DEFAULT_W, DEFAULT_H), thresh: float = DEFAULT_THRESH):
    original_h, original_w = image.shape[:2]
    img_resized = cv2.resize(to_gray(image), (int(target_size[0]), int(target_size[1])))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=-1)  # (H,W,1)

    if handle.kind == "keras":
        x = np.expand_dims(img_input, axis=0)  # (1,H,W,1)
        pred = handle.obj.predict(x, verbose=0)[0].squeeze()
    elif handle.kind == "tflite":
        interpreter = handle.obj
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.expand_dims(img_input, axis=0).astype(input_details[0]["dtype"])  # (1,H,W,1)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index']).squeeze()
    else:
        raise ValueError("Unknown model handle kind")

    mask = (pred > float(thresh)).astype("uint8") * 255
    return cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

# ===============================
# Large-file friendly stack I/O
# ===============================

def save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out_f:
        shutil.copyfileobj(uploaded_file, out_f)
    return tmp_path

def get_stack_info(path: str):
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile as tiff
        with tiff.TiffFile(path) as tif:
            return "tiff", len(tif.pages)
    else:
        with Image.open(path) as im:
            n_frames = getattr(im, "n_frames", 1)
        return "pil", n_frames

def get_frame_as_array(path: str, index: int):
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile as tiff
        with tiff.TiffFile(path) as tif:
            arr = tif.pages[index].asarray()
        return arr
    else:
        with Image.open(path) as im:
            im.seek(index)
            return np.array(im)

# ===============================
# Session state shortcuts
# ===============================

ss = st.session_state
for key, default in [
    ("last_mask", None),
    ("last_processed", None),
    ("last_cropped", None),
    ("last_original", None),
    ("tmp_path", None),
    ("n_frames", 0),
    ("frame_idx", 0),
    ("processed_stack", None),
    ("masks_stack", None),
    ("metrics_stack", None),
    ("result_view_idx", 0),
    ("unet_handle", None),
]:
    if key not in ss:
        ss[key] = default

# ======= Auto-load default model on startup (no interaction) =======

def ensure_default_model_loaded():
    if ss.get("unet_handle") is not None:
        return
    for cand in DEFAULT_MODEL_CANDIDATES:
        if not cand:
            continue
        # SavedModel dir or file
        if os.path.isdir(cand) or os.path.isfile(cand):
            h = load_unet_any(cand)
            if h is not None:
                ss.unet_handle = h
                break

ensure_default_model_loaded()

# ===============================
# TABS
# ===============================

tab_seg, tab_metrics, tab_help = st.tabs(["🧩 Segmenter", "📊 Metrics", "📖 Instructions"])

with tab_seg:
    st.title("🔬 NucleoNet")

    uploaded_file = st.file_uploader(
        "📂 Upload a microscopy image or stack (e.g., multi-page TIFF, GIF)",
        type=["jpg", "png", "tif", "tiff", "bmp", "gif"]
    )
    if uploaded_file:
        ss.tmp_path = save_upload_to_temp(uploaded_file)
        fmt, n_frames = get_stack_info(ss.tmp_path)
        ss.n_frames = n_frames

        if n_frames > 1:
            st.sidebar.header("🎞️ Stack")
            ss.frame_idx = st.sidebar.slider("Frame", 0, n_frames - 1, 0, key="in_frame_slider")
            st.caption(f"Detected stack with **{n_frames}** frames.")
        else:
            ss.frame_idx = 0

        current_img = get_frame_as_array(ss.tmp_path, ss.frame_idx)
        ss.last_original = current_img
        show_u8(current_img, caption=f"📷 Original image (frame {ss.frame_idx+1}/{n_frames})", width=320)

        # ---------------------------
        # Cropping
        # ---------------------------
        st.sidebar.header("✂️ Cropping")
        h_img, w_img = current_img.shape[:2]
        crop_top = st.sidebar.slider("Top", 0, max(1, h_img // 2), 10, key="crop_top")
        crop_bottom = st.sidebar.slider("Bottom", 0, max(1, h_img // 2), 10, key="crop_bottom")
        crop_left = st.sidebar.slider("Left", 0, max(1, w_img // 2), 10, key="crop_left")
        crop_right = st.sidebar.slider("Right", 0, max(1, w_img // 2), 10, key="crop_right")

        def apply_crop(img):
            h, w = img.shape[:2]
            t = min(crop_top, max(0, h - 1))
            b = min(crop_bottom, max(0, h - 1))
            l = min(crop_left, max(0, w - 1))
            r = min(crop_right, max(0, w - 1))
            if t + b < h and l + r < w:
                return img[t:h - b, l:w - r]
            return img

        cropped = apply_crop(current_img)
        ss.last_cropped = cropped

        # ---------------------------
        # Pre-processing
        # ---------------------------
        st.sidebar.header("🧪 Image processing")

        clahe_on = st.sidebar.checkbox("CLAHE", value=False, key="pp_clahe")
        if clahe_on:
            clahe_clip = st.sidebar.slider("CLAHE Clip", 1.0, 10.0, 2.0, key="pp_clahe_clip")
            clahe_grid = st.sidebar.slider("CLAHE Grid", 4, 16, 8, key="pp_clahe_grid")

        bilateral_on = st.sidebar.checkbox("Bilateral filter", value=False, key="pp_bilateral")
        if bilateral_on:
            bf_d = st.sidebar.slider("Diameter", 1, 15, 9, key="pp_bf_d")
            bf_sigmaC = st.sidebar.slider("Sigma Color", 1, 150, 75, key="pp_bf_sigmaC")
            bf_sigmaS = st.sidebar.slider("Sigma Space", 1, 150, 75, key="pp_bf_sigmaS")

        unsharp_on = st.sidebar.checkbox("Unsharp masking", value=False, key="pp_unsharp")
        if unsharp_on:
            us_ksize = st.sidebar.slider("Kernel size", 3, 21, 9, step=2, key="pp_us_ksize")
            us_sigma = st.sidebar.slider("Gaussian sigma", 1.0, 10.0, 3.0, key="pp_us_sigma")

        nlm_on = st.sidebar.checkbox("Denoising (NLM)", value=False, key="pp_nlm")
        if nlm_on:
            nlm_h = st.sidebar.slider("NLM strength (h)", 1, 30, 10, key="pp_nlm_h")

        adapt_th_on = st.sidebar.checkbox("Adaptive threshold (binary)", value=False, key="pp_adapt")
        if adapt_th_on:
            ad_block = st.sidebar.slider("Block size", 3, 51, 11, step=2, key="pp_ad_block")
            ad_C = st.sidebar.slider("Constant C", -20, 20, 2, key="pp_ad_C")

        def apply_preproc(img_in):
            proc_local = img_in.copy()
            if clahe_on:
                proc_local = apply_clahe(proc_local, clahe_clip, clahe_grid)
            if bilateral_on:
                proc_local = apply_bilateral(proc_local, bf_d, bf_sigmaC, bf_sigmaS)
            if unsharp_on:
                proc_local = apply_unsharp(proc_local, us_ksize, us_sigma)
            if nlm_on:
                proc_local = apply_nlm_denoise(proc_local, nlm_h)
            if adapt_th_on:
                proc_local = apply_threshold(proc_local, ad_block, ad_C)
            return proc_local

        proc = apply_preproc(cropped)

        # ---------------------------
        # Segmentation controls → cfg
        # ---------------------------
        st.sidebar.header("🧩 Segmentation")
        segment_option = st.sidebar.selectbox(
            "Segmentation method",
            ["None", "Otsu (binary)", "Adaptive (binary)", "Watershed", "U-Net"],
            key="seg_mode"
        )

        cfg = {"mode": segment_option}

        if segment_option == "Otsu (binary)":
            st.sidebar.subheader("⚙️ Otsu settings")
            cfg["otsu_invert"] = st.sidebar.checkbox("Invert binary", value=False, key="otsu_invert")
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔧 Morphology (pre-split)")
            cfg["morph_op"] = st.sidebar.selectbox("Operation", ["none", "erode", "open", "close", "dilate"], index=0, key="otsu_morph_op")
            cfg["morph_ksize"] = st.sidebar.slider("Kernel size", 1, 31, 3, step=2, key="otsu_morph_ksize")
            cfg["morph_iters"] = st.sidebar.slider("Iterations", 1, 10, 1, key="otsu_morph_iters")

        elif segment_option == "Adaptive (binary)":
            st.sidebar.subheader("⚙️ Adaptive settings")
            cfg["ad_method"] = st.sidebar.selectbox("Method", ["mean", "gaussian"], index=0, key="ad_method")
            cfg["ad_block"] = st.sidebar.slider("Block size", 3, 99, 15, step=2, key="ad_block")
            cfg["ad_C"] = st.sidebar.slider("C (bias)", -20, 20, 2, key="ad_C")
            cfg["ad_invert"] = st.sidebar.checkbox("Invert binary", value=False, key="ad_invert")
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔧 Morphology (pre-split)")
            cfg["morph_op"] = st.sidebar.selectbox("Operation", ["none", "erode", "open", "close", "dilate"], index=0, key="ad_morph_op")
            cfg["morph_ksize"] = st.sidebar.slider("Kernel size", 1, 31, 3, step=2, key="ad_morph_ksize")
            cfg["morph_iters"] = st.sidebar.slider("Iterations", 1, 10, 1, key="ad_morph_iters")

        elif segment_option == "Watershed":
            st.sidebar.subheader("⚙️ Watershed settings")
            cfg["blur_type"] = st.sidebar.selectbox("Blur type", ["median", "gaussian", "none"], index=0, key="ws_blur_type")
            cfg["blur_ksize"] = st.sidebar.slider("Blur kernel size", 1, 31, 5, step=2, key="ws_blur_ksize")
            cfg["blur_sigma"] = st.sidebar.slider("Gaussian sigma (if gaussian)", 0.1, 10.0, 1.0, key="ws_blur_sigma")
            cfg["thresh_mode"] = st.sidebar.selectbox("Threshold mode", ["otsu", "adaptive_mean", "adaptive_gaussian"], index=0, key="ws_thresh_mode")
            cfg["adaptive_block"] = st.sidebar.slider("Adaptive block size", 3, 51, 11, step=2, key="ws_adaptive_block")
            cfg["adaptive_C"] = st.sidebar.slider("Adaptive C", -20, 20, 2, key="ws_adaptive_C")
            cfg["invert_binary"] = st.sidebar.checkbox("Invert binary", value=False, key="ws_invert")
            cfg["morph_kernel"] = st.sidebar.slider("Morph kernel (ellipse)", 1, 21, 3, step=2, key="ws_morph_kernel")
            cfg["opening_iter"] = st.sidebar.slider("Opening iterations", 0, 5, 2, key="ws_opening_iter")
            cfg["dilate_iter"] = st.sidebar.slider("Dilate (sure BG) iterations", 0, 10, 3, key="ws_dilate_iter")
            cfg["dist_type"] = st.sidebar.selectbox("Distance type", ["L2", "L1", "C"], index=0, key="ws_dist_type")
            cfg["dist_mask"] = st.sidebar.selectbox("Distance mask size", [3, 5], index=1, key="ws_dist_mask")
            cfg["sure_fg_frac"] = st.sidebar.slider("Sure FG threshold (× max dist)", 0.0, 1.0, 0.7, 0.01, key="ws_sure_fg")
            cfg["connectivity"] = st.sidebar.selectbox("ConnectedComponents connectivity", [4, 8], index=1, key="ws_conn")
            cfg["show_debug"] = st.sidebar.checkbox("Show intermediate steps", value=False, key="ws_debug")

        elif segment_option == "U-Net":
            st.sidebar.subheader("📦 Load U-Net model")
            # Default: prefer auto-loaded handle if exists
            loaded_path = ss.unet_handle.path if ss.get("unet_handle") else ""
            default_path = loaded_path or ("/mnt/data/model.tflite" if os.path.exists("/mnt/data/model.tflite") else "")
            manual = st.sidebar.text_input("Model path (.keras / .h5 / .tflite / SavedModel dir)", value=default_path)
            cfg["model_path"] = manual
            cfg["unet_thresh"] = st.sidebar.slider("U-Net probability threshold", 0.1, 0.95, float(DEFAULT_THRESH), 0.05, key="unet_thresh")
            st.sidebar.caption(f"Input size (from metadata.json if present): {DEFAULT_W}×{DEFAULT_H}; normalization: {INPUT_NORM}")

            if manual and (not ss.get("unet_handle") or (manual and Path(ss.unet_handle.path).as_posix() != Path(manual).as_posix())):
                ss.unet_handle = load_unet_any(manual)
                if ss.unet_handle is None:
                    st.sidebar.error("Could not load model. Check path/format.")
                else:
                    st.sidebar.success(f"Loaded model: {ss.unet_handle.kind} — {Path(ss.unet_handle.path).name}")

        # Whole stack toggle
        process_all = False
        if n_frames > 1:
            st.sidebar.markdown("---")
            process_all = st.sidebar.checkbox(
                "Process entire stack (all frames)",
                value=False,
                help="Apply the same crop, preprocessing and segmentation to every frame and preview with a slider.",
                key="proc_all_stack"
            )

        # ---------------------------
        # Pure segmentation
        # ---------------------------
        def apply_segmentation_no_ui(proc_img, cfg):
            mode = cfg["mode"]

            if mode == "Otsu (binary)":
                raw_mask = otsu_threshold(proc_img, invert=cfg["otsu_invert"])
                seg_mask = morph_separate(raw_mask, op=cfg["morph_op"], ksize=cfg["morph_ksize"], iters=cfg["morph_iters"])
                return seg_mask, seg_mask

            if mode == "Adaptive (binary)":
                raw_mask = adaptive_local_threshold(
                    proc_img, block=cfg["ad_block"], C=cfg["ad_C"], method=cfg["ad_method"], invert=cfg["ad_invert"]
                )
                seg_mask = morph_separate(raw_mask, op=cfg["morph_op"], ksize=cfg["morph_ksize"], iters=cfg["morph_iters"])
                return seg_mask, seg_mask

            if mode == "Watershed":
                if cfg.get("show_debug", False):
                    ws_overlay, ws_mask, _ = watershed_segment(
                        proc_img,
                        blur_type=cfg["blur_type"], blur_ksize=cfg["blur_ksize"], blur_sigma=cfg["ws_blur_sigma"] if "ws_blur_sigma" in cfg else cfg["blur_sigma"],
                        thresh_mode=cfg["thresh_mode"], adaptive_block=cfg["adaptive_block"], adaptive_C=cfg["adaptive_C"],
                        invert_binary=cfg["invert_binary"], morph_kernel=cfg["morph_kernel"],
                        opening_iter=cfg["opening_iter"], dilate_iter=cfg["dilate_iter"],
                        dist_type=cfg["dist_type"], dist_mask=cfg["dist_mask"],
                        sure_fg_frac=cfg["sure_fg_frac"], connectivity=cfg["connectivity"],
                        return_debug=True
                    )
                else:
                    ws_overlay, ws_mask = watershed_segment(
                        proc_img,
                        blur_type=cfg["blur_type"], blur_ksize=cfg["blur_ksize"], blur_sigma=cfg["blur_sigma"],
                        thresh_mode=cfg["thresh_mode"], adaptive_block=cfg["adaptive_block"], adaptive_C=cfg["adaptive_C"],
                        invert_binary=cfg["invert_binary"], morph_kernel=cfg["morph_kernel"],
                        opening_iter=cfg["opening_iter"], dilate_iter=cfg["dilate_iter"],
                        dist_type=cfg["dist_type"], dist_mask=cfg["dist_mask"],
                        sure_fg_frac=cfg["sure_fg_frac"], connectivity=cfg["connectivity"],
                        return_debug=False
                    )
                return ws_overlay, ws_mask

            if mode == "U-Net":
                handle = ss.get("unet_handle")
                if handle is None:
                    mp = cfg.get("model_path", "")
                    if mp and (os.path.exists(mp)):
                        handle = load_unet_any(mp)
                        ss.unet_handle = handle
                if handle is not None:
                    mask_u = segment_with_unet_handle(handle, proc_img, target_size=(DEFAULT_W, DEFAULT_H), thresh=cfg["unet_thresh"])
                    return mask_u, mask_u
                else:
                    st.warning("U-Net model is not loaded — check the sidebar.")
                    return proc_img, None

            return proc_img, None

        # =========================
        # Single frame vs entire stack
        # =========================
        if not process_all:
            processed_to_show, mask_to_use = apply_segmentation_no_ui(proc, cfg)

            # ---------------------------
            # Post-processing (NEW)
            # ---------------------------
            st.subheader("🧹 Post-processing (improve separation)")
            if mask_to_use is not None:
                with st.expander("Post-processing options", expanded=True):
                    pp_morph_op   = st.selectbox("Morphological operations", ["none", "erode", "open", "close", "dilate"], index=0)
                    pp_morph_ks   = st.slider("Kernel size", 1, 31, 3, step=2)
                    pp_morph_it   = st.slider("Iterations", 1, 10, 1)
                    pp_fill       = st.checkbox("Fill holes", value=True)
                    pp_min_area   = st.slider("Remove small objects (min area, px)", 0, 2000, 0, step=10)
                    pp_ws         = st.checkbox("Watershed split (on mask)", value=False)
                    pp_ws_thresh  = st.slider("Watershed sure-FG fraction", 0.3, 0.95, 0.6, 0.01)
                    pp_open_rec   = st.checkbox("Opening by reconstruction", value=False, help="Requires scikit-image")
                    pp_rec_size   = st.slider("Reconstruction kernel (square)", 1, 31, 5, step=2)
                    pp_skeleton   = st.checkbox("Skeletonize (1px-wide)", value=False, help="Requires scikit-image")

                post_mask = mask_to_use.copy()
                # Morphology (classical)
                post_mask = morph_separate(post_mask, op=pp_morph_op, ksize=pp_morph_ks, iters=pp_morph_it)
                # Fill holes
                if pp_fill:
                    post_mask = fill_holes_u8(post_mask)
                # Remove small objects
                if pp_min_area > 0:
                    _, kept = count_cells_filtered(post_mask, min_area_px=int(pp_min_area), exclude_border=False, do_fill_holes=False)
                    post_mask = kept
                # Watershed split on mask
                if pp_ws:
                    post_mask = watershed_on_mask(post_mask, min_peak_frac=pp_ws_thresh)
                # Opening by reconstruction (shape-preserving)
                if pp_open_rec:
                    post_mask = opening_by_reconstruction(post_mask, pp_rec_size)
                # Skeletonization last
                if pp_skeleton:
                    post_mask = skeletonize_binary(post_mask)

                col1, col2 = st.columns(2)
                with col1:
                    show_u8(mask_to_use, caption="Raw mask", width=320)
                with col2:
                    show_u8(post_mask, caption="Post-processed mask", width=320)

                processed_display = post_mask
                ss.last_mask = post_mask
            else:
                processed_display = processed_to_show
                ss.last_mask = None

            st.subheader("🔍 Preview")
            colp1, colp2 = st.columns(2)
            with colp1:
                show_u8(proc, caption="Preprocessed image", width=320)
            with colp2:
                show_u8(processed_display, caption="Segmentation result", width=320)

            ss.last_processed = processed_display

            # Download single frame
            st.subheader("📥 Download result")
            result_image = convert_to_pil(processed_display)
            buf = BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button("📁 Download PNG",
                               buf.getvalue(),
                               file_name=f"result_frame_{ss.frame_idx+1}.png",
                               mime="image/png")

            # Clear stack caches
            ss.processed_stack = None
            ss.masks_stack = None
            ss.metrics_stack = None
            ss.result_view_idx = 0

        else:
            st.subheader(f"🗂 Processing entire stack ({n_frames} frames)")
            processed_list, masks_list, metrics_list = [], [], []

            default_min_area = max(20, int(0.0005 * (proc.shape[0] * proc.shape[1])))

            # Post-processing settings for stack
            with st.expander("Post-processing options for stack", expanded=True):
                pp_morph_op  = st.selectbox("Morphological op", ["none", "erode", "open", "close", "dilate"], index=0, key="stk_morph_op")
                pp_morph_ks  = st.slider("Kernel size", 1, 31, 3, step=2, key="stk_morph_ks")
                pp_morph_it  = st.slider("Iterations", 1, 10, 1, key="stk_morph_it")
                pp_fill      = st.checkbox("Fill holes", value=True, key="stk_fill")
                pp_min_area  = st.slider("Remove small objects (min area, px)", 0, 5000, 0, step=20, key="stk_min_area")
                pp_ws        = st.checkbox("Watershed split (on mask)", value=False, key="stk_ws")
                pp_ws_thresh = st.slider("Watershed sure-FG fraction", 0.3, 0.95, 0.6, 0.01, key="stk_ws_thr")
                pp_open_rec  = st.checkbox("Opening by reconstruction", value=False, key="stk_open_rec")
                pp_rec_size  = st.slider("Reconstruction kernel (square)", 1, 31, 5, step=2, key="stk_rec_size")
                pp_skeleton  = st.checkbox("Skeletonize (1px-wide)", value=False, key="stk_skeleton")

            for idx in range(n_frames):
                img_i = get_frame_as_array(ss.tmp_path, idx)
                img_i = crop_image(img_i, crop_top, crop_bottom, crop_left, crop_right)
                proc_i = apply_preproc(img_i)
                out_i, mask_i = apply_segmentation_no_ui(proc_i, cfg)

                if mask_i is not None:
                    pm = morph_separate(mask_i, op=pp_morph_op, ksize=pp_morph_ks, iters=pp_morph_it)
                    if pp_fill:
                        pm = fill_holes_u8(pm)
                    if pp_min_area > 0:
                        _, kept = count_cells_filtered(pm, min_area_px=int(pp_min_area), exclude_border=False, do_fill_holes=False)
                        pm = kept
                    if pp_ws:
                        pm = watershed_on_mask(pm, min_peak_frac=pp_ws_thresh)
                    if pp_open_rec:
                        pm = opening_by_reconstruction(pm, pp_rec_size)
                    if pp_skeleton:
                        pm = skeletonize_binary(pm)
                else:
                    pm = out_i

                processed_list.append(pm)
                masks_list.append(pm if mask_i is not None else None)

                if mask_i is not None:
                    occ, total, pct = occupancy_from_mask(pm)
                    n_cells, _ = count_cells_filtered(pm, min_area_px=default_min_area, exclude_border=True, do_fill_holes=True)
                else:
                    occ, total, pct, n_cells = 0, proc_i.size, 0.0, 0
                metrics_list.append({
                    "frame_index": idx,
                    "occupied_pixels": int(occ),
                    "total_pixels": int(total),
                    "coverage_percent": float(pct),
                    "detected_cells": int(n_cells)
                })

            ss.processed_stack = processed_list
            ss.masks_stack = masks_list
            ss.metrics_stack = metrics_list
            ss.result_view_idx = 0

            st.markdown("---")
            st.subheader("👀 Preview processed stack")
            if n_frames > 0:
                ss.result_view_idx = st.slider("Result frame", 0, n_frames - 1, 0, key="result_view_slider")
                show_u8(processed_list[ss.result_view_idx],
                        caption=f"Result frame {ss.result_view_idx+1}/{n_frames}", width=512)

            st.markdown("---")
            st.subheader("📦 Download stack results")

            def encode_stack_to_tiff(images_list):
                pil_imgs = [convert_to_pil(im) for im in images_list]
                buf_tif = BytesIO()
                if len(pil_imgs) == 1:
                    pil_imgs[0].save(buf_tif, format="TIFF")
                else:
                    pil_imgs[0].save(buf_tif, format="TIFF", save_all=True, append_images=pil_imgs[1:])
                return buf_tif.getvalue()

            tiff_bytes = encode_stack_to_tiff(processed_list)
            st.download_button(
                "📁 Download processed stack (TIFF)",
                data=tiff_bytes,
                file_name="processed_stack.tiff",
                mime="image/tiff"
            )

            import pandas as pd
            dfm = pd.DataFrame(metrics_list)
            st.download_button(
                "📊 Download per-frame metrics (CSV)",
                data=dfm.to_csv(index=False).encode("utf-8"),
                file_name="stack_metrics.csv",
                mime="text/csv"
            )

            st.info("Tip: Metrics tab uses the last single-frame run. For stack-wide metrics, use the CSV above.")

with tab_metrics:
    st.title("📊 Metrics")

    if ss.last_mask is not None:
        mask_u8 = normalize_to_u8(ss.last_mask)

        occ, total, pct = occupancy_from_mask(mask_u8)

        st.subheader("Cell counting controls")
        default_min_area = max(20, int(0.0005 * total))
        min_area_px = st.slider("Minimum cell area (px)", 1, max(10000, default_min_area), default_min_area, step=1, key="met_min_area")
        exclude_border = st.checkbox("Exclude cells touching image border", value=True, key="met_excl_border")
        do_fill_holes  = st.checkbox("Fill holes inside cells", value=True, key="met_fill_holes")

        n_cells, kept_mask = count_cells_filtered(
            mask_u8,
            min_area_px=min_area_px,
            exclude_border=exclude_border,
            do_fill_holes=do_fill_holes
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Occupied pixels (mask)", f"{occ:,}")
        c2.metric("Total pixels", f"{total:,}")
        c3.metric("Coverage (mask > 0)", f"{pct:.2f}%")
        c4.metric("Detected cells", f"{n_cells:,}")

        st.caption(
            "Coverage is computed exclusively from the 0/255 binary mask. "
            "Cell count uses connected components after optional hole filling, area filtering, "
            "and border exclusion."
        )

        st.image(kept_mask, caption="Mask used for counting (filtered)", width=320, clamp=True)

        import pandas as pd
        from datetime import datetime
        report_data = {
            "timestamp": [datetime.now().isoformat(timespec="seconds")],
            "frame_index": [ss.get("frame_idx", 0)],
            "occupied_pixels": [occ],
            "total_pixels": [total],
            "coverage_percent": [pct],
            "detected_cells": [n_cells],
            "min_area_px": [int(min_area_px)],
            "exclude_border": [bool(exclude_border)],
            "fill_holes": [bool(do_fill_holes)],
        }
        df_report = pd.DataFrame(report_data)
        st.download_button(
            label="📥 Download metrics (CSV)",
            data=df_report.to_csv(index=False).encode("utf-8"),
            file_name="segmentation_metrics.csv",
            mime="text/csv"
        )
    else:
        st.info("Run a segmentation first to generate a binary mask and compute metrics.")

with tab_help:
    st.title("📖 Instructions")

    # Live model info (if loaded)
    model_info_lines = []
    if ss.get("unet_handle") is not None:
        model_info_lines.append(f"**Loaded model:** `{Path(ss.unet_handle.path).name}`")
        model_info_lines.append(f"**Format:** `{ss.unet_handle.kind}`")
    else:
        if os.path.exists("/mnt/data/model.tflite"):
            model_info_lines.append("**Default model (preferred):** `/mnt/data/model.tflite`")
        if os.path.exists("/mnt/data/model.keras"):
            model_info_lines.append("Also available: `/mnt/data/model.keras`")
    model_info_lines.append(f"**Input size:** {DEFAULT_W}×{DEFAULT_H}")
    model_info_lines.append(f"**Default threshold:** {DEFAULT_THRESH}")
    model_info_lines.append(f"**Normalization:** {INPUT_NORM}")
    model_info_lines.append(f"**Auto-load:** {'ON' if ss.get('unet_handle') else 'OFF'}")

    model_info_md = "\n".join([f"- {line}" for line in model_info_lines])

    st.markdown(f"""
**What this app does**

- Upload a microscopy image **or a stack** (e.g., multi-page TIFF, GIF), optionally apply pre-processing (CLAHE, denoising, unsharp masking, adaptive threshold).
- Segment using configurable **Otsu/Adaptive (binary)**, **Watershed**, or a **U-Net** model.
- The U-Net loader supports **.keras/.h5**, **SavedModel** directory, and **TFLite (.tflite)**.
- Defaults (from `metadata.json` if present): input size, probability threshold, and normalization.
- In **stack mode**, process all frames and preview the **processed stack with a slider**; download the **TIFF stack** and **per-frame CSV metrics**.
- After segmentation, use the **Post-processing** panel to separate touching cells (morphology or watershed), fill holes, remove small objects, **opening by reconstruction**, and **skeletonization**.

**Model info**

{model_info_md}

**Coverage**

Coverage is computed **exclusively from the binary segmentation mask (0/255)**:

\\[
\\text{{coverage}} = \\frac{{\\#(\\text{{mask}} > 0)}}{{\\#(\\text{{all pixels}})}} \\times 100\\%
\\]

**Large files tip**

The app writes uploads to a temporary file and reads **one page at a time**.  
If you see `OSError: [Errno 28] No space left on device`, move Streamlit’s temp directory to a larger drive **before** calling `st.set_page_config`, e.g.:

    import tempfile, os
    tempfile.tempdir = r"D:\\streamlit_temp"
    os.makedirs(tempfile.tempdir, exist_ok=True)
""")
