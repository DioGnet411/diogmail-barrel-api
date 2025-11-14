# ===========================================
# DioGMail Barrel API — Minimal Deployment
# ===========================================

import os
import gdown
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

MODEL_ZIP_URL = "https://drive.google.com/file/d/1AkqV0IyHuXghzFXwGX1nWuSWlZiTLT0H"

# -------------------------------------------
# 1. Download and extract model files
# -------------------------------------------
if not os.path.exists("model/embedder"):
    os.makedirs("model", exist_ok=True)
    print("Downloading DioGMail Barrel weights...")
    gdown.download(MODEL_ZIP_URL, "models.zip", quiet=False)
    os.system("unzip -o models.zip -d model/")
    print("Model extracted.")

# -------------------------------------------
# 2. Load models
# -------------------------------------------
print("Loading embedder...")
embedder = SentenceTransformer("model/embedder")

print("Loading Layer 1 classifier...")
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

sample_dim = logreg.coef_.shape[1]  # should be 384 for MiniLM

barrel = Barrel(sample_dim)
barrel.load_state_dict(torch.load("model/barrel_model.pt", map_location="cpu"))

print("Computing Barrel normalization stats...")
with torch.no_grad():
    dummy = np.zeros((32, sample_dim))
    dummy_t = torch.tensor(dummy, dtype=torch.float32)
    gvals, _ = barrel.ff_goodness(dummy_t)
    barrel.mean_global = gvals.mean()
    barrel.std_global = gvals.std()

# -------------------------------------------
# 3. Email Body Input Format
# -------------------------------------------
class Email(BaseModel):
    subject: str
    body: str
    sender: str = ""

BEST_TH = 0.35

# -------------------------------------------
# 4. API Endpoint
# -------------------------------------------
@app.post("/scan")
def scan_email(e: Email):
    text = e.subject + " [SEP] " + e.body
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
    risk = int((1 - fused) * 100)
    verdict = "legit" if fused >= BEST_TH else "scam"

    return {
        "verdict": verdict,
        "risk": risk,
        "p_lr": round(p_lr, 3),
        "p_barrel": round(p_bar, 3),
        "fused_p_legit": round(fused, 3)
    }
