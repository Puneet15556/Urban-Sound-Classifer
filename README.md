# 🔊 Environmental Sound Classification (UrbanSound8K)

A Streamlit-based web application for **environmental sound classification** using **Mel-frequency features** and an **XGBoost multi-class classifier**, trained on the **UrbanSound8K dataset**.

The app allows users to **upload an audio file (.wav)** and predicts the sound class along with a confidence score.

---

## 🚀 Features

- Upload `.wav` audio files
- Classifies sounds into **10 UrbanSound8K categories**
- Displays:
  - Predicted class
  - Confidence score
- Simple, clean Streamlit UI
- Fast inference using classical ML (XGBoost)

---

## 🧠 Sound Classes (UrbanSound8K)

| Class ID | Class Name |
|--------|------------|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

---

## 🧩 Model & Features

### Feature Extraction
- Mel Spectrogram
- Log-scaled (Mel dB)
- Temporal aggregation:
  - Mean
  - Standard deviation
- Final feature vector size: **128**

### Model
- **XGBoost (multi-class softmax)**
- Supervised learning
- Optimized for tabular audio features

---

## 📂 Project Structure
urban-sound-classifier/
│
├── app.py
├── audio_classifier_xgb.pkl
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧠 Model & Feature Details

### Feature Extraction

- Audio loaded at **22,050 Hz**
- Mel Spectrogram computed
- Converted to **log-scaled Mel (Mel-dB)**
- Temporal aggregation:
  - Mean
  - Standard deviation
- Final feature vector size: **128**

### Model

- **XGBoost multi-class classifier**
- Objective: `multi:softprob`
- Suitable for tabular audio features
- Fast inference and low computational cost

---

## 📊 Dataset

- **UrbanSound8K**
- 8,732 labeled audio clips (≤ 4 seconds)
- 10 urban environmental sound classes
- Publicly available dataset

Dataset link:  
https://urbansounddataset.weebly.com/urbansound8k.html

---

## 📈 Example Output

- **Predicted Class:** dog_bark  
- **Confidence:** 0.87  

The confidence score represents the model’s probability for the predicted class.

---

## 🚧 Future Improvements

- Top-3 class prediction display
- Spectrogram visualization in UI
- CNN-based deep learning comparison
- Real-time audio classification
- Batch audio prediction

---

## 👤 Author

**PUNEET RANJAN**  
Environmental Sound Classification using UrbanSound8K  
(Streamlit • ML • XGBoost)

