# 🎭 Emotion-Based Music Recommender

The **Emotion-Based Music Recommender** is an AI-powered application that detects a user's facial emotion through a webcam and recommends music based on the detected emotion. The system uses computer vision and machine learning techniques to analyze facial expressions in real time and generate personalized music suggestions.

---

## 🚀 Features

- Real-time **facial emotion detection** using webcam  
- Emotion recognition using **MediaPipe and OpenCV**  
- Emotion classification (Happy, Sad, Angry, Neutral, Surprise, etc.)  
- **AI-based music recommendation** based on detected emotion  
- Interactive **Streamlit web interface**  
- Random music suggestions for each emotion category  

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Image processing and face detection  
- **MediaPipe** – Facial landmark detection  
- **NumPy**
- **Streamlit** – Web interface  
- **Machine Learning Concepts**

---

## ⚙️ How It Works

1. The system captures the user's face through a webcam.  
2. Facial landmarks are detected using **MediaPipe Face Mesh**.  
3. The facial features are analyzed to determine the user's emotion.  
4. Based on the detected emotion, the system recommends suitable music.  
5. The recommended songs are displayed in the interface.

---

## 📌 Applications

- Personalized music recommendation systems  
- Mood-based entertainment platforms  
- Mental wellness and stress relief tools  
- Smart entertainment systems  

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
