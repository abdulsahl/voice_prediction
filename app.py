import os
import numpy as np
import pandas as pd
import librosa
import scipy.stats
import joblib
import streamlit as st
from st_audiorec import st_audiorec   # komponen rekam audio di browser

# === FUNGSI EKSTRAKSI ===
def extract_stats(arr):
    return [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        scipy.stats.skew(arr),
        scipy.stats.kurtosis(arr)
    ]

def extract_features(y, sr):
    feat_vector = []

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Raw signal
    feat_vector.extend(extract_stats(y))

    # Temporal
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feat_vector.extend(extract_stats(zcr))

    rms = librosa.feature.rms(y=y)[0]
    feat_vector.extend(extract_stats(rms))

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feat_vector.extend(extract_stats(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feat_vector.extend(extract_stats(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feat_vector.extend(extract_stats(rolloff))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    feat_vector.extend(extract_stats(flatness))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for band in contrast:
        feat_vector.extend(extract_stats(band))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for bin_chroma in chroma:
        feat_vector.extend(extract_stats(bin_chroma))

    # MFCC + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for coeff in mfcc:
        feat_vector.extend(extract_stats(coeff))
    for coeff in delta:
        feat_vector.extend(extract_stats(coeff))
    for coeff in delta2:
        feat_vector.extend(extract_stats(coeff))

    return feat_vector

# === LOAD ARTEFAK ===
X_columns = joblib.load("X_columns.pkl")
selected_features = joblib.load("selected_features.pkl")
pca_model = joblib.load("pca_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# === FUNGSI PREDIKSI ===
def predict_audio(y, sr):
    feat_vector = extract_features(y, sr)
    df_feat = pd.DataFrame([feat_vector], columns=X_columns)

    df_sel = df_feat[selected_features]
    df_pca = pca_model.transform(df_sel)

    pred = svm_model.predict(df_pca)[0]
    probs = svm_model.predict_proba(df_pca)[0]

    return pred, probs

# === STREAMLIT UI ===
st.title("Klasifikasi Suara: Buka / Tutup")

# Upload file dengan preview
uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # preview audio
    y, sr = librosa.load(uploaded_file, sr=None)
    pred, probs = predict_audio(y, sr)
    st.write("Prediksi:", pred)
    st.write("Probabilitas:", dict(zip(svm_model.classes_, probs)))
    st.bar_chart(pd.Series(probs, index=svm_model.classes_))

# Rekam audio langsung di browser
st.subheader("Rekam Audio dari Browser")
wav_audio_data = st_audiorec()   # tombol record muncul di UI

if wav_audio_data is not None:
    # Preview hasil rekaman
    st.audio(wav_audio_data, format="audio/wav")

    # Simpan ke file sementara
    with open("recorded.wav", "wb") as f:
        f.write(wav_audio_data)

    # Load untuk prediksi
    y, sr = librosa.load("recorded.wav", sr=None)
    pred, probs = predict_audio(y, sr)
    st.write("Prediksi:", pred)
    st.write("Probabilitas:", dict(zip(svm_model.classes_, probs)))
    st.bar_chart(pd.Series(probs, index=svm_model.classes_))