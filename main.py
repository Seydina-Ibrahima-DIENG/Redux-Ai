from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from PIL import Image
import uuid
import os

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # mets l’URL de ton front en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Nb-Variables-Initiales", "X-Nb-Composantes-Retenues"],
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

    mean_col = np.mean(matrice, axis=0)
    X_centered = matrice - mean_col

    pca_full = PCA()
    pca_full.fit(X_centered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, n_components_variance) + 1

    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_centered)
    X_reconstructed = pca_final.inverse_transform(X_pca) + mean_col

    signal_final = X_reconstructed.reshape(-1)

    return signal_final, sr, X_centered.shape[1], n_components


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

    pca_full = PCA()
    pca_full.fit(img_centered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, n_components_variance) + 1

    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(img_centered)
    img_reconstructed = (X_pca @ pca_final.components_) + mean
    img_reconstructed = np.clip(img_reconstructed * 255.0, 0, 255).astype(np.uint8)

    return img_reconstructed, nb_variables_initiales, n_components


# =============================
# ENDPOINT PRO
# =============================
@app.post("/traiter-fichier")
async def traiter_fichier(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="Fichier invalide")

    ext = file.filename.split(".")[-1].lower()

    if ext not in ["wav", "png"]:
        raise HTTPException(status_code=400, detail="Format non supporté")

    input_filename = f"input_{uuid.uuid4()}.{ext}"
    output_filename = f"output_{uuid.uuid4()}.{ext}"

    # Sauvegarde temporaire
    with open(input_filename, "wb") as f:
        f.write(await file.read())

    try:
        if ext == "wav":
            signal_final, sr, nb_init, nb_retenues = traiter_audio_fichier(input_filename)
            sf.write(output_filename, signal_final, sr)
            media_type = "audio/wav"

        elif ext == "png":
            img_final, nb_init, nb_retenues = traiter_image_fichier(input_filename)
            Image.fromarray(img_final).save(output_filename)
            media_type = "image/png"

    finally:
        if os.path.exists(input_filename):
            os.remove(input_filename)

    # Supprimer le fichier de sortie après envoi
    background = BackgroundTask(lambda: os.remove(output_filename))

    return FileResponse(
        path=output_filename,
        media_type=media_type,
        filename=f"processed_{file.filename}",
        headers={
            "X-Nb-Variables-Initiales": str(nb_init),
            "X-Nb-Composantes-Retenues": str(nb_retenues),
        },
        background=background,
    )
