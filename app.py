# ===========================================
# DioGMail Barrel API — Minimal & Stable
# ===========================================

import os
import zipfile
import torch
import joblib
import numpy as np
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ---------------------------------------------------
# Google Drive ZIP containing:
# /embedder (MiniLM)
# logreg.pkl
# barrel_model.pt
# ---------------------------------------------------
MODEL_ZIP_URL = "https://drive.google.com/uc?id=1TTMdcG0Lo7yPA2OJyq-jqczSaUcgznxr"
MODEL_DIR = "model"
EMBEDDER_DIR = "model/embedder"
ZIP_PATH = "models.zip"


# ---------------------------------------------------
# 1. Download + Extract Models (Render-safe)
# ---------------------------------------------------
def ensure_models():
    if os.path.exists(EMBEDDER_DIR) and \
       os.path.exists(f"{MODEL_DIR}/logreg.pkl") and \
       os.path.exists(f"{MODEL_DIR}/barrel_model.pt"):
        print("✅ Models already exist. Skipping download.")
        return

    print("⏳ Downloading model bundle...")
    gdown.download(MODEL_ZIP_URL, ZIP_PATH, quiet=False)

    print("📦 Extracting models...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print("📁 Extracted content:", os.listdir(MODEL_DIR))

    # Safety check
    required = ["embedder", "logreg.pkl", "barrel_model.pt"]
    for item in required:
        if not os.path.exists(f"{MODEL_DIR}/{item}"):
            raise FileNotFoundError(f"❌ Missing: {item}")

    print("✅ All model files extracted successfully.")


ensure_models()


# ---------------------------------------------------
# 2. Load Models
# ---------------------------------------------------
print("🔍 Loading embedder...")
embedder = SentenceTransformer(EMBEDDER_DIR)

print("🔍 Loading Layer-1 classifier...")
logreg_path = f"{MODEL_DIR}/logreg.pkl"
logreg = joblib.load(logreg_path)

print("🔍 Loading Barrel model...")


class Barrel(torch.nn.Module):
    def __init__(self, input_dim, h1=128, h2=64):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, h1)
        self.l2 = torch.nn.Linear(h1, h2)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(1, 1)
        self.mean_global = None
        self.std_global = None

    def ff_goodness(self, x):
        a = self.relu(self.l1(x))
        b = self.relu(self.l2(a))
        g = (b ** 2).mean(dim=1, keepdim=True)
        return g, b

    def forward(self, x):
        g, _ = self.ff_goodness(x)
        g_norm = (g - self.mean_global) / (self.std_global + 1e-9)
        return self.out(g_norm).squeeze(1)


sample_dim = logreg.coef_.shape[1]

barrel = Barrel(sample_dim)
barrel.load_state_dict(torch.load(f"{MODEL_DIR}/barrel_model.pt", map_location="cpu"))

# Compute normalization statistics
with torch.no_grad():
    dummy = np.zeros((32, sample_dim))
    dummy_t = torch.tensor(dummy, dtype=torch.float32)
    gvals, _ = barrel.ff_goodness(dummy_t)
    barrel.mean_global = gvals.mean()
    barrel.std_global = gvals.std()


# ---------------------------------------------------
# 3. Schema
# ---------------------------------------------------
class Email(BaseModel):
    subject: str
    body: str
    sender: str = ""


BEST_TH = 0.35


# ---------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------
@app.get("/")
def root():
    return {"message": "DioGMail Barrel API is running."}


@app.post("/scan")
def scan_email(e: Email):
    text = f"{e.subject} [SEP] {e.body}"
    emb = embedder.encode([text])

    # Layer 1 probability
    p_lr = float(logreg.predict_proba(emb)[0][1])

    # Barrel probability
    with torch.no_grad():
        x_t = torch.tensor(emb, dtype=torch.float32)
        logits = barrel(x_t)
        p_bar = float(torch.sigmoid(logits))

    # Fusion
    fused = (p_lr + p_bar) / 2
    verdict = "legit" if fused >= BEST_TH else "scam"
    risk = int((1 - fused) * 100)

    return {
        "verdict": verdict,
        "risk": risk,
        "p_lr": round(p_lr, 3),
        "p_barrel": round(p_bar, 3),
        "fused_p_legit": round(fused, 3)
    }

