# ===========================================
# DioGMail Barrel API — Minimal Deployment
# ===========================================

import os
import gdown
import zipfile
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

MODEL_ZIP_URL = "https://drive.google.com/uc?id=1TTMdcG0Lo7yPA2OJyq-jqczSaUcgznxr"

# -------------------------------------------------------
# 1. Download and extract model files
# -------------------------------------------------------
if not os.path.exists("model/embedder"):
    os.makedirs("model", exist_ok=True)

    print("⏳ Downloading DioGMail Barrel models from Google Drive...")
    gdown.download(MODEL_ZIP_URL, "models.zip", quiet=False)

    print("📦 Extracting models.zip...")
    with zipfile.ZipFile("models.zip", 'r') as zip_ref:
        zip_ref.extractall("model/")
    print("✅ Models extracted.")

# -------------------------------------------------------
# 2. Load models
# -------------------------------------------------------
print("Loading embedder...")
embedder = SentenceTransformer("model/embedder")

print("Loading logistic layer...")
logreg = joblib.load("model/logreg.pkl")

print("Loading Barrel model...")

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
barrel.load_state_dict(torch.load("model/barrel_model.pt", map_location="cpu"))

print("Computing Barrel normalization stats...")
with torch.no_grad():
    dummy = torch.zeros((32, sample_dim), dtype=torch.float32)
    gvals, _ = barrel.ff_goodness(dummy)
    barrel.mean_global = gvals.mean()
    barrel.std_global = gvals.std()

# -------------------------------------------------------
# 3. Email format
# -------------------------------------------------------
class Email(BaseModel):
    subject: str
    body: str
    sender: str = ""

BEST_TH = 0.35

# -------------------------------------------------------
# 4. Endpoints
# -------------------------------------------------------
@app.get("/")
def home():
    return {"message": "DioGMail Barrel API is running"}

@app.post("/scan")
def scan_email(e: Email):
    text = e.subject + " [SEP] " + e.body
    emb = embedder.encode([text])

    p_lr = float(logreg.predict_proba(emb)[0][1])

    with torch.no_grad():
        logits = barrel(torch.tensor(emb, dtype=torch.float32))
        p_bar = float(torch.sigmoid(logits))

    fused = (p_lr + p_bar) / 2
    risk = int((1 - fused) * 100)
    verdict = "legit" if fused >= BEST_TH else "scam"

    return {
        "verdict": verdict,
        "risk": risk,
        "p_lr": round(p_lr, 3),
        "p_barrel": round(p_bar, 3),
        "fused_p_legit": round(fused, 3)
    }

