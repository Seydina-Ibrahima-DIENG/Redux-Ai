from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
import os
import uuid

app = FastAPI()

# -------------------------------
# Autoriser CORS pour ton front
# -------------------------------
origins = [
    "http://localhost:5500",   # si tu testes en local
    "http://127.0.0.1:5500",
    "https://ton-domaine-frontend.com",  # si tu héberges ton front
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ou ["*"] pour tout autoriser (pas recommandé en prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 1️⃣ Charger audio
# =============================
def charger_audio(chemin_audio):
    signal, sample_rate = sf.read(chemin_audio)

    if len(signal.shape) == 2:
        signal_mono = signal[:, 0]
    else:
        signal_mono = signal

    return signal_mono, sample_rate

# =============================
# 2️⃣ Découper en fenêtres
# =============================
def decouper_en_fenetres(signal_mono, window_size=1024):
    num_windows = len(signal_mono) // window_size
    signal_tronque = signal_mono[:num_windows * window_size]
    matrice = signal_tronque.reshape(num_windows, window_size)
    return matrice

# =============================
# 3️⃣ Appliquer ACP
# =============================
def appliquer_acp_audio(matrice, n_composantes=10):
    mean_col = np.mean(matrice, axis=0)
    X_centered = matrice - mean_col

    pca = PCA(n_components=n_composantes)
    X_pca = pca.fit_transform(X_centered)

    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed += mean_col

    return X_reconstructed

# =============================
# 4️⃣ Reconstruire signal
# =============================
def reconstruire_signal(X_reconstructed):
    return X_reconstructed.reshape(-1)

# =============================
# 🎧 ENDPOINT
# =============================
@app.post("/traiter-audio")
async def traiter_audio(file: UploadFile = File(...)):

    # Vérifier extension
    if not file.filename.endswith(".wav"):
        return {"error": "Seuls les fichiers .wav sont autorisés"}

    # Nom unique
    input_filename = f"input_{uuid.uuid4()}.wav"
    output_filename = f"output_{uuid.uuid4()}.wav"

    # Sauvegarder temporairement
    with open(input_filename, "wb") as buffer:
        buffer.write(await file.read())

    # Traitement
    signal_mono, sr = charger_audio(input_filename)
    X = decouper_en_fenetres(signal_mono)
    X_reconstruit = appliquer_acp_audio(X, n_composantes=5)
    signal_final = reconstruire_signal(X_reconstruit)

    sf.write(output_filename, signal_final, sr)

    # Supprimer input
    os.remove(input_filename)

    # Retourner fichier
    return FileResponse(
        path=output_filename,
        media_type="audio/wav",
        filename="audio_reconstruit.wav"
    )
