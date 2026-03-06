import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import requests
import random
import time
from collections import Counter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Emotion Music AI",
    page_icon="🎭",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>🎭 Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)

# --------------------------------------------------
# LANGUAGE SELECTOR
# --------------------------------------------------
language = st.sidebar.selectbox(
    "🌍 Select Language",
    ["English", "Hindi"]
)

language_keywords = {
    "English": {"suffix": "pop dance hits", "country": "US"},
    "Hindi": {"suffix": "Bollywood songs", "country": "IN"}
}

# Improved mood-based queries
emotion_query_map = {
    "happy": "upbeat dance party energetic",
    "sad": "acoustic slow heartbreak emotional",
    "surprise": "edm remix energetic festival",
    "neutral": "lofi chill study instrumental"
}

emotion_emojis = {
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

emotion_colors = {
    "happy": "#FFF9C4",
    "sad": "#BBDEFB",
    "surprise": "#FFE0B2",
    "neutral": "#E0E0E0"
}

# --------------------------------------------------
# SESSION STATES
# --------------------------------------------------
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = None

if "songs" not in st.session_state:
    st.session_state.songs = []

if "emotion_buffer" not in st.session_state:
    st.session_state.emotion_buffer = []

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# EMOTION DETECTION FUNCTION
# --------------------------------------------------
def detect_emotion_from_landmarks(landmarks, w, h):

    def get_point(index):
        return np.array([landmarks[index].x * w,
                         landmarks[index].y * h])

    left_lip = get_point(61)
    right_lip = get_point(291)
    top_lip = get_point(13)
    bottom_lip = get_point(14)
    chin = get_point(152)
    forehead = get_point(10)

    mouth_width = np.linalg.norm(left_lip - right_lip)
    mouth_height = np.linalg.norm(top_lip - bottom_lip)
    face_height = np.linalg.norm(chin - forehead)

    width_ratio = mouth_width / face_height
    height_ratio = mouth_height / face_height

    if height_ratio > 0.065:
        return "surprise", min(height_ratio * 8, 1.0)

    elif width_ratio > 0.33:
        return "happy", min(width_ratio * 2, 1.0)

    elif width_ratio < 0.32 and height_ratio < 0.03:
        return "sad", 0.75

    else:
        return "neutral", 0.6



# --------------------------------------------------
# CAMERA SECTION
# --------------------------------------------------
col1, col2 = st.columns([2, 1])
run = st.sidebar.checkbox("🎥 Start Camera")
FRAME_WINDOW = col1.image([])

if run:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time() 

    while run and (time.time() - start_time < 5):
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break

        frame = cv2.resize(frame, (480, 360))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1

        if frame_count % 3 == 0:
            results = face_mesh.process(frame_rgb)

            detected_emotion = "neutral"
            confidence = 0.5

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    detected_emotion, confidence = detect_emotion_from_landmarks(
                        face_landmarks.landmark, w, h
                    )

            # Emotion smoothing (last 5 frames)
            st.session_state.emotion_buffer.append(detected_emotion)
            if len(st.session_state.emotion_buffer) > 5:
                st.session_state.emotion_buffer.pop(0)

            final_emotion = Counter(
                st.session_state.emotion_buffer
            ).most_common(1)[0][0]

            # Fetch new songs when emotion changes
            if final_emotion != st.session_state.current_emotion:

                st.session_state.current_emotion = final_emotion

                lang_data = language_keywords[language]

                # Add randomness to prevent same songs
                random_number = random.randint(1, 1000)

                search_term = (
                    emotion_query_map[final_emotion]
                    + " "
                    + lang_data["suffix"]
                    + " "
                    + str(random_number)
                )

                params = {
                    "term": search_term,
                    "media": "music",
                    "entity": "song",
                    "limit": 5,
                    "country": lang_data["country"]
                }

                try:
                    response = requests.get(
                        "https://itunes.apple.com/search",
                        params=params,
                        timeout=5
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.songs = data.get("results", [])[:5]
                    else:
                        st.session_state.songs = []

                except:
                    st.session_state.songs = []

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

# --------------------------------------------------
# DYNAMIC BACKGROUND
# --------------------------------------------------
if st.session_state.current_emotion:
    bg_color = emotion_colors[st.session_state.current_emotion]

    st.markdown(
        f"<style>body {{ background-color: {bg_color}; }}</style>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# EMOTION DISPLAY
# --------------------------------------------------
if st.session_state.current_emotion:
    emoji = emotion_emojis[st.session_state.current_emotion]

    col2.markdown(f"""
    <div style="padding:20px;border-radius:15px;text-align:center;font-size:22px;">
        {emoji} <b>{st.session_state.current_emotion.upper()}</b>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# SONG DISPLAY
# --------------------------------------------------
if st.session_state.songs:
    st.subheader(f"🎵 Recommended {language} Songs")

    playlist_text = ""

    for track in st.session_state.songs:
        artwork = track.get("artworkUrl100", "")
        preview = track.get("previewUrl", "")
        link = track.get("trackViewUrl", "")

        playlist_text += track.get('trackName', '') + " - " + track.get('artistName', '') + "\n"

        st.markdown(f"""
        <div style="background:#1c1f26;padding:15px;border-radius:15px;">
            <img src="{artwork}" width="80"><br>
            🎶 <b>{track.get('trackName','')}</b><br>
            👤 {track.get('artistName','')}
        </div>
        """, unsafe_allow_html=True)

        if preview:
            st.audio(preview)

        st.markdown(f"[Open in iTunes]({link})")
        st.write("---")

    st.download_button("📥 Download Playlist", playlist_text)