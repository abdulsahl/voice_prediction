import streamlit as st
import numpy as np
import librosa
import io
import joblib
from st_audiorec import st_audiorec

# --- KONFIGURASI DAN PARAMETER (TETAP SAMA) ---
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 50
MODEL_PATH = "svm_audio_classifier.joblib"

# --- FUNGSI-FUNGSI UTAMA YANG TELAH DIPERBAIKI ---

@st.cache_resource
def load_model():
    """Memuat pipeline model dari file joblib."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error memuat model di '{MODEL_PATH}': {e}")
        return None

def get_features_from_audio_bytes(wav_bytes):
    """
    FUNGSI KUNCI: Pipeline lengkap dari byte audio ke fitur siap prediksi.
    Ini adalah satu-satunya fungsi yang kita butuhkan untuk pemrosesan.
    """
    try:
        # 1. Muat audio langsung dari byte menggunakan librosa.
        #    Ini menggantikan pydub dan konversi manual yang rawan kesalahan.
        #    librosa secara otomatis menangani konversi ke float.
        audio_stream = io.BytesIO(wav_bytes)
        y, sr = librosa.load(audio_stream, sr=SAMPLE_RATE)

        # 2. Trim keheningan (sama seperti di notebook)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # 3. Ekstrak fitur MFCC (sama seperti di notebook)
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=N_MFCC)
        
        # 4. Padding/Truncating (sama seperti di notebook)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        # 5. Ratakan (flatten) fitur untuk input model SVM
        mfcc_flattened = mfcc.reshape(1, -1)
        
        return mfcc_flattened

    except Exception as e:
        st.error(f"Error saat memproses audio: {e}")
        return None

# --- TAMPILAN APLIKASI STREAMLIT ---

st.set_page_config(page_title="Klasifikasi Suara (SVM)", layout="wide")
st.title("🎤 Aplikasi Klasifikasi Suara: 'Buka' vs 'Tutup'")
st.subheader("Model: Support Vector Machine (SVM)")
st.markdown("---")

model = load_model()

if model is not None:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1. Rekam Suara Anda")
        wav_audio_data = st_audiorec()

    with col2:
        st.subheader("2. Hasil Analisis")
        if wav_audio_data is not None and len(wav_audio_data) > 1000:
            st.audio(wav_audio_data, format='audio/wav')
            
            with st.spinner('Menganalisis suara...'):
                # Panggil fungsi pipeline tunggal kita
                features = get_features_from_audio_bytes(wav_audio_data)
                
                if features is not None:
                    # Lakukan prediksi. Pipeline akan menangani scaling.
                    prediction = model.predict(features)[0]
                    prediction_proba = model.predict_proba(features)
                    
                    label = "Tutup" if prediction == 1 else "Buka"
                    confidence = prediction_proba[0][prediction]
                    
                    st.success(f"**Prediksi:** `{label}`")
                    st.info(f"**Tingkat Keyakinan:** `{confidence:.2%}`")
        else:
            st.info("Menunggu rekaman suara untuk dianalisis...")
else:
    st.warning("Model `svm_audio_classifier.joblib` tidak ditemukan.")