import joblib
import librosa
import numpy as np
import streamlit as st
import tempfile


model = joblib.load("audio_classifier_xgb.pkl")

class_map = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

def extract_mel(file_path, n_mels=64):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels =n_mels , n_fft=1024 , hop_length=512)
    mel_db = librosa.power_to_db(mel , ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)  # fixed-size vector
    mel_std = np.std(mel_db, axis=1)
    features = np.concatenate([mel_mean, mel_std])
    features = (features - np.mean(features)) / np.std(features)
    # mel_db = (mel_db - mel_mean) / mel_std
    return features
    
    
def predict_audio(file_path, model):
    features = extract_mel(file_path)   # SAME function as training

    features = features.reshape(1, -1)  # (1, 128)

    probs = model.predict_proba(features)
    pred_id = probs.argmax(axis=1)[0]
    pred_class = class_map[pred_id]
    return pred_class , probs[0] , pred_id

    
st.title("🔊 Environmental Sound Classifier")
st.write("UrbanSound8K | XGBoost + Mel Features")

uploaded_file = st.file_uploader(
    "Upload an audio file (.wav)",
    type=["wav"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(uploaded_file.read())
    audio_path = tmp.name

    st.audio(audio_path)

    predicted_class , probs , pred_id = predict_audio(file_path=audio_path , model=model)
    confidence = probs[pred_id]



    st.markdown(f"## 🧠 Prediction")
    st.markdown(f"### **Class:** {predicted_class}")
    if confidence > 0.6:
        st.success(f"###### **Confidence:** {round(float(confidence), 3)}")
    else:
        st.warning(f"Low Confidence")    

