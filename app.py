import os
import json
from typing import Any, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

# ====================== Config ======================
MODEL_DIR = os.getenv("MODEL_DIR", "models")
SAVEDMODEL_PATH = os.path.join(MODEL_DIR, "saved_model")   # carpeta
H5_PATH = os.path.join(MODEL_DIR, "model.h5")              # archivo
INDEX_PATH = os.getenv("INDEX_PATH", "data/index.json")
DEFAULT_INPUT_SIZE = int(os.getenv("INPUT_SIZE", "64"))    # si el modelo no define tamaño
SCORE_PRECISION = 6

# ====================== App =========================
app = FastAPI(title="Flower Classifier (TF/Keras)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # ajusta en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== Etiquetas ======================
def load_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # {"0": "daisy", ...} -> ordénalo por índice
        return [label for _, label in sorted(data.items(), key=lambda kv: int(kv[0]))]
    if isinstance(data, list):
        return data
    return []

LABELS: List[str] = load_labels(INDEX_PATH)

# ============== Cargar modelo TF/Keras ==============
def load_keras_model() -> Optional[tf.keras.Model]:
    # Prioridad: SavedModel/  > model.h5
    if os.path.isdir(SAVEDMODEL_PATH):
        try:
            m = tf.keras.models.load_model(SAVEDMODEL_PATH)
            print(f"[INFO] Loaded SavedModel: {SAVEDMODEL_PATH}")
            return m
        except Exception as e:
            print(f"[WARN] Can't load SavedModel: {e}")
    if os.path.exists(H5_PATH):
        try:
            m = tf.keras.models.load_model(H5_PATH)
            print(f"[INFO] Loaded H5 model: {H5_PATH}")
            return m
        except Exception as e:
            print(f"[WARN] Can't load H5: {e}")
    return None

MODEL: Optional[tf.keras.Model] = load_keras_model()

# Detectar layout esperado por el modelo (channels_last vs channels_first)
def detect_expected_layout(model: tf.keras.Model) -> Tuple[str, Tuple[Optional[int], Optional[int], Optional[int]]]:
    """
    Devuelve ("NHWC" o "NCHW", (H, W, C_or_NONE)).
    Si no se puede inferir, asumimos NHWC (HWC).
    """
    try:
        # Para Keras: model.inputs[0].shape típicamente es (None, H, W, C) o (None, C, H, W)
        ishape = model.inputs[0].shape
        if len(ishape) == 4:
            n, d1, d2, d3 = ishape  # TensorShape
            dims = [int(d) if d is not None else None for d in (d1, d2, d3)]
            # Heurística: si C está al final -> NHWC
            # Si la primera dim después de batch es 3/1 y hay 3 canales -> NCHW
            if d3 is not None and (int(d3) in (1, 3) or (LABELS and int(d3) == len(LABELS))):
                return "NHWC", (dims[0], dims[1], dims[2])
            if d1 is not None and int(d1) in (1, 3):
                return "NCHW", (dims[1], dims[2], dims[0])  # guardamos como (H, W, C) equivalente
    except Exception:
        pass
    # fallback
    return "NHWC", (None, None, None)

EXPECTED_LAYOUT, _ = (("NHWC", (None, None, None)) if MODEL is None
                      else detect_expected_layout(MODEL))

def get_input_size() -> int:
    # Intenta inferir H y W del modelo
    if MODEL is not None:
        try:
            ishape = MODEL.inputs[0].shape
            if len(ishape) == 4:
                # (None, H, W, C) o (None, C, H, W)
                dims = [d if d is not None else None for d in ishape]
                if EXPECTED_LAYOUT == "NHWC":
                    h = int(dims[1]) if dims[1] is not None else DEFAULT_INPUT_SIZE
                else:  # NCHW
                    h = int(dims[2]) if dims[2] is not None else DEFAULT_INPUT_SIZE
                return h
        except Exception:
            pass
    return DEFAULT_INPUT_SIZE

INPUT_SIZE = get_input_size()

# =================== Utils comunes ==================
class PredictPayload(BaseModel):
    instances: List[Any]

def ensure_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

def hwc_to_nhwc(x: np.ndarray) -> np.ndarray:
    # (H, W, C) -> (1, H, W, C)
    return x[None, :, :, :]

def hwc_to_nchw(x: np.ndarray) -> np.ndarray:
    # (H, W, C) -> (1, C, H, W)
    return np.transpose(x, (2, 0, 1))[None, :, :, :]

def preprocess_hwc_64(img_arr_hwc: np.ndarray) -> np.ndarray:
    """Recibe HxWxC (RGB), redimensiona a INPUT_SIZE y normaliza [0..1]."""
    from PIL import Image
    pil = Image.fromarray(img_arr_hwc.astype(np.uint8)) if img_arr_hwc.dtype != np.float32 else Image.fromarray((img_arr_hwc * 255).astype(np.uint8))
    pil = pil.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(pil, dtype=np.float32)
    arr = ensure_float01(arr)
    return arr  # HWC

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = ensure_float01(arr)  # HWC
    return arr

def adapt_to_model_input(hwc_arr: np.ndarray) -> np.ndarray:
    """Devuelve batch en el layout que el modelo espera (NHWC o NCHW)."""
    if EXPECTED_LAYOUT == "NHWC":
        return hwc_to_nhwc(hwc_arr)
    else:
        return hwc_to_nchw(hwc_arr)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

def label_of(i: int) -> str:
    if 0 <= i < len(LABELS):
        return str(LABELS[i])
    return str(i)

def model_predict(batch: np.ndarray) -> np.ndarray:
    """
    Ejecuta el modelo y devuelve logits o probabilidades (N, num_classes).
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    # Para TF/Keras, preferimos predict() en CPU
    preds = MODEL(batch, training=False).numpy() if callable(MODEL) else MODEL.predict(batch)
    return preds

# ===================== Endpoints ====================
@app.get("/healthz")
def healthz():
    return {
        "model_loaded": MODEL is not None,
        "expected_layout": EXPECTED_LAYOUT,
        "input_size": INPUT_SIZE,
        "labels": len(LABELS),
        "paths": {"saved_model": os.path.isdir(SAVEDMODEL_PATH), "h5": os.path.exists(H5_PATH)},
    }

@app.get("/v1/models/flower:metadata")
def metadata():
    return {
        "framework": "tensorflow-keras",
        "expected_layout": EXPECTED_LAYOUT,
        "input": {"size": INPUT_SIZE, "dtype": "float32", "range": "[0,1]"},
        "labels": {"count": len(LABELS), "sample": LABELS[:5]},
        "endpoints": ["/v1/models/flower:predict", "/v1/models/flower:predict-image"]
    }

@app.post("/v1/models/flower:predict")
def predict(payload: PredictPayload):
    if not payload.instances:
        raise HTTPException(status_code=400, detail="Falta 'instances'.")

    # Acepta HxWxC o 1xHxWxC o 1xCxHxW
    arr = np.asarray(payload.instances[0], dtype=np.float32)
    if arr.ndim == 3:
        # HWC -> resize/normalize -> adapt
        if arr.shape[-1] != 3:
            raise HTTPException(status_code=400, detail="Se espera HxWx3 (RGB).")
        arr = ensure_float01(arr)
        if arr.shape[0] != INPUT_SIZE or arr.shape[1] != INPUT_SIZE:
            arr = preprocess_hwc_64(arr)
        batch = adapt_to_model_input(arr)
    elif arr.ndim == 4:
        # Asumimos batch ya dado; normalizamos y, si hace falta, transponemos
        arr = ensure_float01(arr)
        # Intento mínimo de auto-arreglo si dims no coinciden
        if EXPECTED_LAYOUT == "NHWC" and arr.shape[1] in (1, 3):  # viene NCHW
            arr = np.transpose(arr, (0, 2, 3, 1))
        if EXPECTED_LAYOUT == "NCHW" and arr.shape[-1] in (1, 3):  # viene NHWC
            arr = np.transpose(arr, (0, 3, 1, 2))
        batch = arr
    else:
        raise HTTPException(status_code=400, detail="Forma no soportada. Use HxWxC, 1xHxWxC o 1xCxHxW.")

    logits = model_predict(batch)
    # Normaliza a probas por si el modelo devuelve logits
    probs = softmax(logits, axis=1)
    top_idx = int(np.argmax(probs, axis=1)[0])
    top_score = float(np.round(probs[0, top_idx], SCORE_PRECISION))
    top_label = label_of(top_idx)

    return {
        "predictions": np.round(probs, SCORE_PRECISION).tolist(),
        "top": {"index": top_idx, "score": top_score, "label": top_label}
    }

@app.post("/v1/models/flower:predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")

    hwc = preprocess_pil(img)       # HWC float32 [0..1], size=INPUT_SIZE
    batch = adapt_to_model_input(hwc)
    logits = model_predict(batch)
    probs = softmax(logits, axis=1)
    top_idx = int(np.argmax(probs, axis=1)[0])
    top_score = float(np.round(probs[0, top_idx], SCORE_PRECISION))
    top_label = label_of(top_idx)

    return {
        "predictions": np.round(probs, SCORE_PRECISION).tolist(),
        "top": {"index": top_idx, "score": top_score, "label": top_label}
    }
