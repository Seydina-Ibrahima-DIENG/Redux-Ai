from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from PIL import Image
import uuid
import os

app = FastAPI()

# -------------------------------
# Autoriser CORS pour le front
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux préciser ton front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 🔹 AUDIO
# =============================
def traiter_audio_fichier(input_path, n_components_variance=0.9):
    signal, sr = sf.read(input_path)
    if signal.ndim == 2:
        signal = signal[:, 0]

    window_size = 1024
    num_windows = len(signal) // window_size
    signal_tronque = signal[:num_windows * window_size]
    matrice = signal_tronque.reshape(num_windows, window_size)

    # Centrer
    mean_col = np.mean(matrice, axis=0)
    X_centered = matrice - mean_col

    # ACP complète pour calcul du nombre de composantes
    pca_full = PCA()
    pca_full.fit(X_centered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, n_components_variance) + 1

    # ACP finale
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_centered)
    X_reconstructed = pca_final.inverse_transform(X_pca) + mean_col

    signal_final = X_reconstructed.reshape(-1)

    return signal_final, sr, X_centered.shape[1], n_components  # (nb variables initiales, nb retenues)

# =============================
# 🔹 IMAGE
# =============================
def traiter_image_fichier(input_path, n_components_variance=0.9):
    img_pil = Image.open(input_path).convert("L")
    img_array = np.array(img_pil, dtype=float) / 255.0

    mean = np.mean(img_array, axis=1, keepdims=True)
    img_centered = img_array - mean

    h, w = img_array.shape
    nb_variables_initiales = w

    # ACP complète
    pca_full = PCA()
    pca_full.fit(img_centered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, n_components_variance) + 1

    # ACP finale
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(img_centered)
    img_reconstructed = (X_pca @ pca_final.components_) + mean
    img_reconstructed = np.clip(img_reconstructed * 255.0, 0, 255).astype(np.uint8)

    return img_reconstructed, nb_variables_initiales, n_components

# =============================
# ENDPOINT
# =============================
@app.post("/traiter-fichier")
async def traiter_fichier(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    input_filename = f"input_{uuid.uuid4()}.{ext}"
    output_filename = f"output_{uuid.uuid4()}.{ext}"

    # Sauvegarder fichier temporaire
    with open(input_filename, "wb") as f:
        f.write(await file.read())

    try:
        if ext == "wav":
            signal_final, sr, nb_init, nb_retenues = traiter_audio_fichier(input_filename, 0.9)
            sf.write(output_filename, signal_final, sr)
        elif ext == "png":
            img_final, nb_init, nb_retenues = traiter_image_fichier(input_filename, 0.9)
            img_pil = Image.fromarray(img_final)
            img_pil.save(output_filename)
        else:
            return JSONResponse({"error": "Seuls les fichiers .wav et .png sont autorisés"}, status_code=400)
    finally:
        os.remove(input_filename)

    # Retourner fichier + stats
    return {
        "file": f"processed_{file.filename}",
        "nb_variables_initiales": nb_init,
        "nb_composantes_retenues": nb_retenues
    }
