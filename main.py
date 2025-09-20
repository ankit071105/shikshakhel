import streamlit as st
import cv2
import numpy as np
import sqlite3
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import pygame
import speech_recognition as sr
import time
import random
from gtts import gTTS
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import BytesIO
import json
import torch
import torchvision
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from deepface import DeepFace
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# Initialize database with schema checking
def init_db():
    conn = sqlite3.connect('education_platform.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if c.fetchone() is None:
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, progress INTEGER,
                     total_score INTEGER, created_date DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    else:
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        if 'total_score' not in columns:
            c.execute(
                "ALTER TABLE users ADD COLUMN total_score INTEGER DEFAULT 0")
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='game_scores'")
    if c.fetchone() is None:
        c.execute('''CREATE TABLE game_scores
                     (id INTEGER PRIMARY KEY, user_id INTEGER, game_name TEXT,
                     score INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='learning_analytics'")
    if c.fetchone() is None:
        c.execute('''CREATE TABLE learning_analytics
                     (id INTEGER PRIMARY KEY, user_id INTEGER, game_name TEXT,
                     accuracy FLOAT, time_taken FLOAT, difficulty INTEGER,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_stories'")
    if c.fetchone() is None:
        c.execute('''CREATE TABLE ai_stories
                     (id INTEGER PRIMARY KEY, user_id INTEGER, prompt TEXT,
                     story_text TEXT, image_url TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()
@st.cache_resource
def load_yolo_model():
    try:
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path='hand_detection.pt',
            force_reload=True)
        return model
    except BaseException:
        st.warning(
            "Custom hand detection model not found. Using pre-trained YOLOv5s (may be less accurate for hands).")
        model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True)
        return model
@st.cache_resource
def init_face_detection():
    mtcnn = MTCNN(keep_all=True, device='cpu')
    return mtcnn

def load_emotion_model():
    pass

# Initialize OpenCV-based hand detection
@st.cache_resource
def init_opencv_hands():
    # Initialize OpenCV background subtractor for motion detection
    backSub = cv2.createBackgroundSubtractorMOG2()
    return backSub

# Simple hand detection using contour analysis
def detect_hand_contours(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            hand_contours.append(contour)
    
    return hand_contours

# Detect hand position using background subtraction
def detect_hand_position(frame, backSub):
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            hand_positions.append((cx, cy, w, h))
    
    return hand_positions, fgMask

# Main application
def main():
    st.set_page_config(
        page_title="AI Learning Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_db()

    # Load models
    if 'yolo_model' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.yolo_model = load_yolo_model()
            st.session_state.mtcnn = init_face_detection()
            st.session_state.backSub = init_opencv_hands()

    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .game-card {
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px 0 rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        color: white;
        font-weight: bold;
    }
    .score-board {
        background: linear-gradient(135deg, #43CBFF 0%, #9708CC 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    @keyframes fall {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    .falling-letter {
        animation: fall 0.5s ease-out;
        font-size: 24px;
        font-weight: bold;
        position: absolute;
    }
    .pulse {
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

    # App header
    st.markdown(
        '<h1 class="main-header">🤖 AI-Powered Learning Platform</h1>',
        unsafe_allow_html=True)

    # Sidebar for user profile
    with st.sidebar:
        st.header("👤 User Profile")
        user_name = st.text_input("Enter your name")
        user_age = st.slider("Select your age", 5, 15, 10)

        if st.button("Start Learning Adventure"):
            conn = sqlite3.connect('education_platform.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (name, age, progress, total_score) VALUES (?, ?, ?, ?)",
                (user_name,
                 user_age,
                 0,
                 0))
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            st.session_state.user = {
                "name": user_name,
                "age": user_age,
                "id": user_id}
            st.success(f"Welcome {user_name}! Let's start learning.")

    # Main content area
    if 'user' in st.session_state:
        # Display scoreboard
        conn = sqlite3.connect('education_platform.db')
        user_scores = pd.read_sql(
            "SELECT * FROM game_scores WHERE user_id = ?",
            conn,
            params=(
                st.session_state.user['id'],
            ))
        user_data = pd.read_sql("SELECT * FROM users WHERE id = ?",
                                conn, params=(st.session_state.user['id'],))
        conn.close()

        total_score = user_data['total_score'].iloc[0] if not user_data.empty else 0

        st.markdown(f"""
        <div class="score-board">
            <h2>🏆 Score: {total_score}</h2>
            <p>Level: {total_score // 100 + 1}</p>
        </div>
        """, unsafe_allow_html=True)

        # Game selection
        st.subheader("🎮 Choose Your Learning Adventure:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🐍 Alphabet Snake Game")
            st.write("Catch alphabets with hand gestures and voice commands")
            if st.button("Play Snake Game", key="snake"):
                st.session_state.game = "alphabet_snake"
                # Reset game state when switching games
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🎯 Color Target Game")
            st.write("Point at colored targets with your finger")
            if st.button("Play Target Game", key="target"):
                st.session_state.game = "color_target"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🎤 Voice Command Challenge")
            st.write("Control the game with your voice commands")
            if st.button("Play Voice Game", key="voice"):
                st.session_state.game = "voice_challenge"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        # Add more games
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("📊 Learning Dashboard")
            st.write("View your progress with AI-powered analytics")
            if st.button("View Dashboard", key="dashboard"):
                st.session_state.game = "learning_dashboard"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col5:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("😊 Emotion Learning")
            st.write("Learn about emotions with facial expressions")
            if st.button("Play Emotion Game", key="emotion"):
                st.session_state.game = "emotion_learning"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col6:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🧠 AI Math Challenge")
            st.write("Solve adaptive math problems with AI")
            if st.button("Play Math Game", key="math"):
                st.session_state.game = "ai_math"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        # New games row
        st.subheader("🎯 More Learning Games")

        col7, col8, col9 = st.columns(3)

        with col7:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🔺 Geometry Gesture Game")
            st.write("Make shapes with your hands to learn geometry")
            if st.button("Play Geometry Game", key="geometry"):
                st.session_state.game = "geometry_gesture"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col8:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🧪 Science Lab Simulator")
            st.write("Perform virtual science experiments with hand gestures")
            if st.button("Play Science Game", key="science"):
                st.session_state.game = "science_lab"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col9:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🔤 English Word Builder")
            st.write("Form words with hand gestures and voice commands")
            if st.button("Play English Game", key="english"):
                st.session_state.game = "english_word"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        # Additional games
        col10, col11, col12 = st.columns(3)

        with col10:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🎨 Drawing Recognition")
            st.write("Draw shapes and objects for the AI to recognize")
            if st.button("Play Drawing Game", key="drawing"):
                st.session_state.game = "drawing_recognition"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col11:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🎵 Music Rhythm Game")
            st.write("Clap to the rhythm and create music patterns")
            if st.button("Play Music Game", key="music"):
                st.session_state.game = "music_rhythm"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        with col12:
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            st.subheader("🧩 Puzzle Solver")
            st.write("Solve puzzles using hand gestures and voice commands")
            if st.button("Play Puzzle Game", key="puzzle"):
                st.session_state.game = "puzzle_solver"
                if 'camera' in st.session_state and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
            st.markdown('</div>', unsafe_allow_html=True)

        # Game area
        if 'game' in st.session_state:
            st.markdown("---")
            if st.session_state.game == "alphabet_snake":
                alphabet_snake_game()
            elif st.session_state.game == "color_target":
                color_target_game()
            elif st.session_state.game == "voice_challenge":
                voice_challenge_game()
            elif st.session_state.game == "learning_dashboard":
                learning_dashboard()
            elif st.session_state.game == "emotion_learning":
                emotion_learning_game()
            elif st.session_state.game == "ai_math":
                ai_math_game()
            elif st.session_state.game == "geometry_gesture":
                geometry_gesture_game()
            elif st.session_state.game == "science_lab":
                science_lab_game()
            elif st.session_state.game == "english_word":
                english_word_game()
            elif st.session_state.game == "drawing_recognition":
                drawing_recognition_game()
            elif st.session_state.game == "music_rhythm":
                music_rhythm_game()
            elif st.session_state.game == "puzzle_solver":
                puzzle_solver_game()
    else:
        st.info(
            "Please enter your name and age in the sidebar to start your learning adventure!")

def alphabet_snake_game():
    st.header("🐍 Alphabet Snake Game")

    st.write("""
    Catch the falling alphabets with your hand!
    Move your hand to control the basket and say the alphabet you catch.
    Make sure your hand is clearly visible against a contrasting background.
    """)

    # Initialize game state
    if 'snake_score' not in st.session_state:
        st.session_state.snake_score = 0
        st.session_state.falling_letters = []
        st.session_state.last_letter_time = time.time()
        st.session_state.basket_pos = 0.5  # Center position
        st.session_state.caught_letters = []
        st.session_state.voice_feedback = ""
        st.session_state.letter_animations = {}

    # Display score
    st.subheader(f"Score: {st.session_state.snake_score}")

    # Game area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Camera feed
        run_camera = st.checkbox("Start Camera", key="snake_cam")

        if run_camera:
            if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
                st.session_state.camera = cv2.VideoCapture(0)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            FRAME_WINDOW = st.image([])

            # Generate falling letters
            current_time = time.time()
            if current_time - st.session_state.last_letter_time > 2:  # New letter every 2 seconds
                # Random uppercase letter
                new_letter = chr(random.randint(65, 90))
                letter_id = f"{new_letter}_{int(time.time() * 1000)}"
                st.session_state.falling_letters.append(
                    {
                        'id': letter_id, 'letter': new_letter, 'x': random.uniform(
                            0.1, 0.9), 'y': 0, 'speed': random.uniform(
                            0.01, 0.03), 'color': (
                            random.randint(
                                0, 255), random.randint(
                                0, 255), random.randint(
                                0, 255))})
                st.session_state.letter_animations[letter_id] = {
                    "start_time": time.time(), "caught": False}
                st.session_state.last_letter_time = current_time

            # Process camera feed for hand tracking using OpenCV
            def process_frame(frame):
                # Use OpenCV for hand detection
                hand_positions, fgMask = detect_hand_position(frame, st.session_state.backSub)
                
                # Get hand positions
                if hand_positions:
                    # Use the largest detected hand
                    largest_hand = max(hand_positions, key=lambda pos: pos[2] * pos[3])
                    cx = largest_hand[0] / frame.shape[1]
                    st.session_state.basket_pos = cx

                    # Draw bounding box
                    x, y, w, h = largest_hand
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Hand", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw basket
                basket_width = 100
                basket_x = int(
                    st.session_state.basket_pos *
                    frame.shape[1] -
                    basket_width /
                    2)
                basket_y = frame.shape[0] - 50

                cv2.rectangle(frame,
                              (basket_x, basket_y),
                              (basket_x + basket_width, basket_y + 20),
                              (0, 255, 0), -1)

                # Update and draw falling letters with animations
                letters_to_remove = []
                for i, letter_data in enumerate(
                        st.session_state.falling_letters):
                    letter_data['y'] += letter_data['speed']
                    letter_x = int(letter_data['x'] * frame.shape[1])
                    letter_y = int(letter_data['y'] * frame.shape[0])

                    # Draw letter with color and animation effect
                    if not st.session_state.letter_animations[letter_data['id']]["caught"]:
                        cv2.putText(
                            frame,
                            letter_data['letter'],
                            (letter_x,
                             letter_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            letter_data['color'],
                            3)
                        cv2.putText(
                            frame,
                            letter_data['letter'],
                            (letter_x,
                             letter_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,
                             255,
                             255),
                            1)

                    # Check if letter is caught
                    if (basket_y <= letter_y <= basket_y + 20 and
                        basket_x <= letter_x <= basket_x + basket_width and
                            not st.session_state.letter_animations[letter_data['id']]["caught"]):
                        st.session_state.letter_animations[letter_data['id']
                                                           ]["caught"] = True
                        st.session_state.letter_animations[letter_data['id']]["catch_time"] = time.time(
                        )
                        st.session_state.caught_letters.append(
                            letter_data['letter'])
                        st.session_state.snake_score += 10

                current_time = time.time()
                st.session_state.falling_letters = [
                    l for l in st.session_state.falling_letters
                    if l['y'] < 1.2 and (
                        not st.session_state.letter_animations[l['id']]["caught"] or
                        current_time - st.session_state.letter_animations[l['id']]["catch_time"] < 1.0
                    )
                ]

                return frame

            while run_camera and st.session_state.camera.isOpened():
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                frame = process_frame(frame)

                # Convert to RGB for Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
        else:
            if 'camera' in st.session_state and st.session_state.camera.isOpened():
                st.session_state.camera.release()

    with col2:
        st.subheader("Caught Letters")
        if st.session_state.caught_letters:
            st.write("Say these letters:")
            # Show last 5 letters
            for letter in st.session_state.caught_letters[-5:]:
                st.markdown(
                    f"<h2 style='text-align: center; color: {random.choice(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFBE0B'])};'>{letter}</h2>",
                    unsafe_allow_html=True)

        # Voice recognition for caught letters
        if st.button("🎤 Say the Letters", key="say_letters"):
            with st.spinner("Listening..."):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    try:
                        recognizer.adjust_for_ambient_noise(
                            source, duration=0.5)
                        audio = recognizer.listen(source, timeout=3)
                        spoken_text = recognizer.recognize_google(
                            audio).upper()

                        # Check if spoken letters match caught letters
                        correct_count = 0
                        for letter in st.session_state.caught_letters:
                            if letter in spoken_text:
                                correct_count += 1

                        if correct_count > 0:
                            st.session_state.snake_score += correct_count * 5
                            st.session_state.voice_feedback = f"Correct! +{correct_count * 5} points"
                            st.balloons()
                        else:
                            st.session_state.voice_feedback = "Try again! Say the letters you caught."

                        # Clear caught letters after voice attempt
                        st.session_state.caught_letters = []

                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        st.session_state.voice_feedback = "Couldn't understand. Try again!"
                    except Exception as e:
                        st.session_state.voice_feedback = f"Error: {e}"

            st.write(st.session_state.voice_feedback)

        if st.button("🔄 Reset Game", key="reset_snake"):
            st.session_state.snake_score = 0
            st.session_state.falling_letters = []
            st.session_state.last_letter_time = time.time()
            st.session_state.caught_letters = []
            st.session_state.letter_animations = {}
            st.experimental_rerun()

        if st.button("💾 Save Score", key="save_snake"):
            conn = sqlite3.connect('education_platform.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
                (st.session_state.user['id'],
                 "Alphabet Snake",
                 st.session_state.snake_score))
            c.execute(
                "UPDATE users SET total_score = total_score + ? WHERE id = ?",
                (st.session_state.snake_score,
                 st.session_state.user['id']))
            conn.commit()
            conn.close()
            st.success("Score saved successfully!")

# Color Target Game with OpenCV-based hand detection
def color_target_game():
    st.header("🎯 Color Target Game")

    st.write("""
    Point at colored targets with your finger to pop them!
    Try to hit as many targets as you can in 30 seconds.
    """)

    # Initialize game state
    if 'target_score' not in st.session_state:
        st.session_state.target_score = 0
        st.session_state.targets = []
        st.session_state.game_start_time = time.time()
        st.session_state.game_duration = 30  # 30 seconds game
        st.session_state.last_target_time = time.time()
        st.session_state.target_animations = {}

    # Display score and time
    current_time = time.time()
    time_left = max(0, st.session_state.game_duration -
                    (current_time - st.session_state.game_start_time))

    st.subheader(
        f"Score: {st.session_state.target_score} | Time: {int(time_left)}s")

    # Game area
    run_camera = st.checkbox("Start Camera", key="target_cam")

    if run_camera:
        if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FRAME_WINDOW = st.image([])

        # Generate targets
        if time_left > 0 and current_time - st.session_state.last_target_time > 1:
            target_id = f"target_{int(time.time() * 1000)}"
            st.session_state.targets.append(
                {
                    'id': target_id, 'x': random.uniform(
                        0.1, 0.9), 'y': random.uniform(
                        0.1, 0.9), 'size': random.randint(
                        30, 50), 'color': (
                        random.randint(
                            0, 255), random.randint(
                                0, 255), random.randint(
                                    0, 255)), 'points': random.randint(
                                        5, 20)})
            st.session_state.target_animations[target_id] = {
                "start_time": time.time(), "hit": False}
            st.session_state.last_target_time = current_time

        # Process camera feed for hand tracking using OpenCV
        def process_target_frame(frame):
            # Use OpenCV for hand detection
            hand_positions, fgMask = detect_hand_position(frame, st.session_state.backSub)

            # Draw targets and check for hits with animations
            targets_to_remove = []
            for i, target in enumerate(st.session_state.targets):
                x_pos = int(target['x'] * frame.shape[1])
                y_pos = int(target['y'] * frame.shape[0])

                # Draw target with animation if hit
                if not st.session_state.target_animations[target['id']]["hit"]:
                    cv2.circle(
                        frame, (x_pos, y_pos), target['size'], target['color'], -1)
                    cv2.circle(
                        frame, (x_pos, y_pos), target['size'], (255, 255, 255), 2)

                    # Draw points value
                    cv2.putText(frame,
                                str(target['points']),
                                (x_pos - 10,
                                 y_pos + 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255,
                                    255,
                                    255),
                                2)

                # Check if hand hits target
                if hand_positions and not st.session_state.target_animations[target['id']]["hit"]:
                    for hand_pos in hand_positions:
                        x, y, w, h = hand_pos
                        distance = np.sqrt(
                            (x - x_pos)**2 + (y - y_pos)**2)
                        if distance < target['size']:
                            st.session_state.target_animations[target['id']
                                                               ]["hit"] = True
                            st.session_state.target_animations[target['id']]["hit_time"] = time.time(
                            )
                            st.session_state.target_score += target['points']

            # Remove hit targets after animation duration
            current_time = time.time()
            st.session_state.targets = [
                t for t in st.session_state.targets
                if not st.session_state.target_animations[t['id']]["hit"] or
                current_time - st.session_state.target_animations[t['id']]["hit_time"] < 0.5
            ]

            return frame

        while run_camera and st.session_state.camera.isOpened() and time_left > 0:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = process_target_frame(frame)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            # Update time
            current_time = time.time()
            time_left = max(0, st.session_state.game_duration -
                            (current_time - st.session_state.game_start_time))
    else:
        if 'camera' in st.session_state and st.session_state.camera.isOpened():
            st.session_state.camera.release()

    if time_left <= 0:
        st.success(f"Game Over! Final Score: {st.session_state.target_score}")
        st.balloons()

    if st.button("🔄 New Game", key="new_target_game"):
        st.session_state.target_score = 0
        st.session_state.targets = []
        st.session_state.game_start_time = time.time()
        st.session_state.last_target_time = time.time()
        st.session_state.target_animations = {}
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_target"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Color Target",
             st.session_state.target_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.target_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Voice Challenge Game with improved UI
def voice_challenge_game():
    st.header("🎤 Voice Command Challenge")

    st.write("""
    Control the game with your voice!
    Say commands like "left", "right", "up", "down", etc.
    """)

    # Initialize game state
    if 'voice_score' not in st.session_state:
        st.session_state.voice_score = 0
        st.session_state.voice_command = ""
        st.session_state.character_pos = [0.5, 0.5]  # x, y position
        st.session_state.coins = []
        st.session_state.last_coin_time = time.time()
        st.session_state.coin_animations = {}

    # Display score
    st.subheader(f"Score: {st.session_state.voice_score}")

    # Game area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create game canvas with background
        game_canvas = np.ones((400, 600, 3), dtype=np.uint8) * 200

        # Draw grid background
        for i in range(0, game_canvas.shape[1], 50):
            cv2.line(game_canvas, (i, 0),
                     (i, game_canvas.shape[0]), (220, 220, 220), 1)
        for i in range(0, game_canvas.shape[0], 50):
            cv2.line(game_canvas, (0, i),
                     (game_canvas.shape[1], i), (220, 220, 220), 1)

        # Draw character
        char_x = int(st.session_state.character_pos[0] * game_canvas.shape[1])
        char_y = int(st.session_state.character_pos[1] * game_canvas.shape[0])
        cv2.circle(game_canvas, (char_x, char_y), 20, (0, 0, 255), -1)
        cv2.circle(game_canvas, (char_x, char_y), 20, (255, 255, 255), 2)

        # Generate coins
        current_time = time.time()
        if current_time - st.session_state.last_coin_time > 3:  # New coin every 3 seconds
            coin_id = f"coin_{int(time.time() * 1000)}"
            st.session_state.coins.append(
                {
                    'id': coin_id, 'x': random.uniform(
                        0.1, 0.9), 'y': random.uniform(
                        0.1, 0.9), 'value': random.randint(
                        5, 15)})
            st.session_state.coin_animations[coin_id] = {
                "start_time": time.time(), "collected": False}
            st.session_state.last_coin_time = current_time

        # Draw coins with animations
        for coin in st.session_state.coins:
            coin_x = int(coin['x'] * game_canvas.shape[1])
            coin_y = int(coin['y'] * game_canvas.shape[0])

            if not st.session_state.coin_animations[coin['id']]["collected"]:
                # Draw coin with pulsing animation
                time_diff = current_time - \
                    st.session_state.coin_animations[coin['id']]["start_time"]
                size = 15 + int(5 * np.sin(time_diff * 5))
                cv2.circle(game_canvas, (coin_x, coin_y), size, (255, 215, 0), -1)
                cv2.circle(game_canvas, (coin_x, coin_y), size, (255, 255, 255), 2)

                # Draw coin value
                cv2.putText(game_canvas,
                            str(coin['value']),
                            (coin_x - 5,
                             coin_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,
                                0,
                                0),
                            1)

        # Check for coin collection
        for coin in st.session_state.coins:
            if not st.session_state.coin_animations[coin['id']]["collected"]:
                coin_x = int(coin['x'] * game_canvas.shape[1])
                coin_y = int(coin['y'] * game_canvas.shape[0])
                distance = np.sqrt(
                    (char_x - coin_x)**2 + (char_y - coin_y)**2)
                if distance < 35:  # Character + coin radius
                    st.session_state.coin_animations[coin['id']
                                                     ]["collected"] = True
                    st.session_state.coin_animations[coin['id']]["collect_time"] = time.time(
                    )
                    st.session_state.voice_score += coin['value']

        # Remove collected coins after animation
        current_time = time.time()
        st.session_state.coins = [
            c for c in st.session_state.coins
            if not st.session_state.coin_animations[c['id']]["collected"] or
            current_time - st.session_state.coin_animations[c['id']]["collect_time"] < 0.5
        ]

        # Display game canvas
        st.image(game_canvas, channels="BGR", use_column_width=True)

    with col2:
        st.subheader("Voice Commands")
        st.write("Try saying:")
        st.write("- 'left', 'right', 'up', 'down'")
        st.write("- 'move left', 'go right', etc.")
        st.write("- 'collect coin', 'get coin'")

        if st.button("🎤 Start Listening", key="listen_voice"):
            with st.spinner("Listening..."):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    try:
                        recognizer.adjust_for_ambient_noise(
                            source, duration=0.5)
                        audio = recognizer.listen(source, timeout=3)
                        st.session_state.voice_command = recognizer.recognize_google(
                            audio).lower()

                        # Process voice command
                        command = st.session_state.voice_command
                        move_speed = 0.1

                        if "left" in command:
                            st.session_state.character_pos[0] = max(
                                0.1, st.session_state.character_pos[0] - move_speed)
                        if "right" in command:
                            st.session_state.character_pos[0] = min(
                                0.9, st.session_state.character_pos[0] + move_speed)
                        if "up" in command:
                            st.session_state.character_pos[1] = max(
                                0.1, st.session_state.character_pos[1] - move_speed)
                        if "down" in command:
                            st.session_state.character_pos[1] = min(
                                0.9, st.session_state.character_pos[1] + move_speed)

                        st.success(f"Heard: {command}")

                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        st.error("Couldn't understand. Try again!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.write(f"Last command: {st.session_state.voice_command}")

        if st.button("🔄 Reset Game", key="reset_voice"):
            st.session_state.voice_score = 0
            st.session_state.character_pos = [0.5, 0.5]
            st.session_state.coins = []
            st.session_state.last_coin_time = time.time()
            st.session_state.coin_animations = {}
            st.experimental_rerun()

        if st.button("💾 Save Score", key="save_voice"):
            conn = sqlite3.connect('education_platform.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
                (st.session_state.user['id'],
                 "Voice Challenge",
                 st.session_state.voice_score))
            c.execute(
                "UPDATE users SET total_score = total_score + ? WHERE id = ?",
                (st.session_state.voice_score,
                 st.session_state.user['id']))
            conn.commit()
            conn.close()
            st.success("Score saved successfully!")

# Learning Dashboard with advanced analytics
def learning_dashboard():
    st.header("📊 Learning Dashboard")

    conn = sqlite3.connect('education_platform.db')
    user_scores = pd.read_sql(
        "SELECT * FROM game_scores WHERE user_id = ?",
        conn,
        params=(
            st.session_state.user['id'],
        ))
    user_data = pd.read_sql("SELECT * FROM users WHERE id = ?",
                            conn, params=(st.session_state.user['id'],))
    conn.close()

    if not user_scores.empty:
        # Overall progress
        total_score = user_data['total_score'].iloc[0]
        level = total_score // 100 + 1
        progress = total_score % 100

        st.subheader("Overall Progress")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", total_score)
        with col2:
            st.metric("Level", level)
        with col3:
            st.metric("Progress to Next Level", f"{progress}%")

        # Progress bar
        st.progress(progress / 100)

        # Game performance
        st.subheader("Game Performance")
        game_stats = user_scores.groupby('game_name').agg(
            {'score': ['mean', 'max', 'count']}).round(2)
        game_stats.columns = ['Average Score', 'High Score', 'Games Played']
        st.dataframe(game_stats)

        # Score trend
        user_scores['timestamp'] = pd.to_datetime(user_scores['timestamp'])
        user_scores['date'] = user_scores['timestamp'].dt.date

        fig = px.line(user_scores, x='date', y='score',
                      color='game_name', title='Score Trend Over Time')
        st.plotly_chart(fig)

        # Performance by game
        fig2 = px.box(user_scores, x='game_name', y='score',
                      title='Score Distribution by Game')
        st.plotly_chart(fig2)

        # Achievement badges
        st.subheader("🏆 Achievements")
        achievements = []

        if total_score >= 100:
            achievements.append("🌟 First Steps (100 points)")
        if total_score >= 500:
            achievements.append("🚀 Learning Star (500 points)")
        if len(user_scores) >= 10:
            achievements.append("🎮 Game Master (10 games played)")
        if game_stats['High Score'].max() >= 50:
            achievements.append("💯 High Scorer (50+ in a game)")

        if achievements:
            for achievement in achievements:
                st.success(achievement)
        else:
            st.info("Keep playing to earn achievements!")

    else:
        st.info("Play some games to see your learning analytics!")
def emotion_learning_game():
    st.header("😊 Emotion Learning")

    st.write("""
    Learn about emotions through facial expressions!
    Make different facial expressions and see if the AI can recognize them.
    """)

    # Initialize game state
    if 'emotion_score' not in st.session_state:
        st.session_state.emotion_score = 0
        st.session_state.current_emotion = random.choice(
            ["happy", "sad", "surprised", "angry", "neutral"])
        st.session_state.detected_emotion = ""
        st.session_state.last_detection_time = 0
        st.session_state.cooldown = 5  # seconds between detections

    st.subheader(f"Score: {st.session_state.emotion_score}")
    st.markdown(
        f"<h2 style='text-align: center; color: #FF6B6B;'>Show me: {st.session_state.current_emotion.upper()}</h2>",
        unsafe_allow_html=True)

    run_camera = st.checkbox("Start Camera", key="emotion_cam")

    if run_camera:
        if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FRAME_WINDOW = st.image([])

        def process_emotion_frame(frame):
            current_time = time.time()

            # Only analyze every few seconds to reduce computation
            if current_time - st.session_state.last_detection_time > st.session_state.cooldown:
                try:
                    # Convert frame to RGB for DeepFace
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Analyze emotion using DeepFace
                    analysis = DeepFace.analyze(
                        rgb_frame,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    
                    if analysis and len(analysis) > 0:
                        emotion_result = analysis[0]['dominant_emotion']
                        st.session_state.detected_emotion = emotion_result
                        
                        # Check if detected emotion matches target
                        if emotion_result == st.session_state.current_emotion:
                            st.session_state.emotion_score += 20
                            st.session_state.current_emotion = random.choice(
                                ["happy", "sad", "surprised", "angry", "neutral"])
                            st.balloons()
                
                except Exception as e:
                    st.error(f"Emotion detection error: {e}")
                
                st.session_state.last_detection_time = current_time

            # Draw emotion text on frame
            cv2.putText(frame,
                        f"Detected: {st.session_state.detected_emotion}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            return frame

        while run_camera and st.session_state.camera.isOpened():
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = process_emotion_frame(frame)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    else:
        if 'camera' in st.session_state and st.session_state.camera.isOpened():
            st.session_state.camera.release()

    st.write("### How to Play:")
    st.write("- Look at the emotion you need to express")
    st.write("- Make that facial expression clearly")
    st.write("- The AI will try to detect your emotion")
    st.write("- Get points for correct expressions!")

    if st.button("🔄 New Emotion", key="new_emotion"):
        st.session_state.current_emotion = random.choice(
            ["happy", "sad", "surprised", "angry", "neutral"])
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_emotion"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Emotion Learning",
             st.session_state.emotion_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.emotion_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# AI Math Challenge Game
def ai_math_game():
    st.header("🧠 AI Math Challenge")

    st.write("""
    Solve math problems with adaptive difficulty!
    The AI will adjust the problems based on your performance.
    """)

    # Initialize game state
    if 'math_score' not in st.session_state:
        st.session_state.math_score = 0
        st.session_state.math_level = 1
        st.session_state.current_problem = None
        st.session_state.user_answer = ""
        st.session_state.correct_answer = None
        st.session_state.feedback = ""
        st.session_state.problems_solved = 0
        st.session_state.generate_problem()

    # Problem generation function
    def generate_math_problem(level):
        if level == 1:
            # Easy: Addition and subtraction
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            operation = random.choice(["+", "-"])
            if operation == "+":
                answer = a + b
            else:
                # Ensure positive result
                a, b = max(a, b), min(a, b)
                answer = a - b
            return f"{a} {operation} {b} = ?", answer

        elif level == 2:
            # Medium: Multiplication and division
            if random.choice([True, False]):
                # Multiplication
                a = random.randint(1, 10)
                b = random.randint(1, 10)
                return f"{a} × {b} = ?", a * b
            else:
                # Division (ensure integer result)
                b = random.randint(1, 10)
                answer = random.randint(1, 10)
                a = b * answer
                return f"{a} ÷ {b} = ?", answer

        else:
            # Hard: Mixed operations
            operations = ["+", "-", "×", "÷"]
            num_count = random.randint(3, 4)
            numbers = [random.randint(1, 15) for _ in range(num_count)]
            ops = [random.choice(operations) for _ in range(num_count - 1)]

            # Build expression and calculate answer
            expression = f"{numbers[0]}"
            result = numbers[0]

            for i in range(num_count - 1):
                expression += f" {ops[i]} {numbers[i+1]}"
                if ops[i] == "+":
                    result += numbers[i+1]
                elif ops[i] == "-":
                    result -= numbers[i+1]
                elif ops[i] == "×":
                    result *= numbers[i+1]
                elif ops[i] == "÷":
                    result /= numbers[i+1]

            return f"{expression} = ?", int(result)

    # Generate problem if needed
    if st.session_state.current_problem is None:
        st.session_state.current_problem, st.session_state.correct_answer = generate_math_problem(
            st.session_state.math_level)

    st.subheader(f"Score: {st.session_state.math_score} | Level: {st.session_state.math_level}")

    # Display current problem
    st.markdown(
        f"<h2 style='text-align: center; color: #4CAF50; font-size: 2.5rem;'>{st.session_state.current_problem}</h2>",
        unsafe_allow_html=True)

    # Answer input
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Your Answer:", key="math_answer")
        if st.button("Submit Answer", key="submit_math"):
            try:
                user_answer = int(user_input)
                if user_answer == st.session_state.correct_answer:
                    st.session_state.math_score += st.session_state.math_level * 10
                    st.session_state.problems_solved += 1
                    st.session_state.feedback = "✅ Correct! Great job!"
                    st.balloons()

                    # Level up after 5 correct problems
                    if st.session_state.problems_solved % 5 == 0:
                        st.session_state.math_level = min(
                            3, st.session_state.math_level + 1)
                        st.session_state.feedback += f" 🚀 Level up! Now at level {st.session_state.math_level}"

                else:
                    st.session_state.feedback = f"❌ Incorrect. The answer was {st.session_state.correct_answer}"

                # Generate new problem
                st.session_state.current_problem, st.session_state.correct_answer = generate_math_problem(
                    st.session_state.math_level)
                st.session_state.user_answer = ""

            except ValueError:
                st.session_state.feedback = "Please enter a valid number"

        st.write(st.session_state.feedback)

    with col2:
        st.write("### Tips:")
        st.write("- Take your time to think")
        st.write("- Use paper if needed")
        st.write("- Level increases every 5 correct answers")

    if st.button("🔄 New Problem", key="new_math_problem"):
        st.session_state.current_problem, st.session_state.correct_answer = generate_math_problem(
            st.session_state.math_level)
        st.session_state.feedback = ""
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_math"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Math Challenge",
             st.session_state.math_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.math_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Geometry Gesture Game
def geometry_gesture_game():
    st.header("🔺 Geometry Gesture Game")

    st.write("""
    Make shapes with your hands to learn geometry!
    Try to form different geometric shapes with your fingers.
    """)

    # Initialize game state
    if 'geometry_score' not in st.session_state:
        st.session_state.geometry_score = 0
        st.session_state.target_shape = random.choice(
            ["circle", "triangle", "square", "star"])
        st.session_state.detected_shape = ""
        st.session_state.last_shape_time = 0

    st.subheader(f"Score: {st.session_state.geometry_score}")
    st.markdown(
        f"<h2 style='text-align: center; color: #FF6B6B;'>Make a: {st.session_state.target_shape.upper()}</h2>",
        unsafe_allow_html=True)

    run_camera = st.checkbox("Start Camera", key="geometry_cam")

    if run_camera:
        if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FRAME_WINDOW = st.image([])

        def process_geometry_frame(frame):
            current_time = time.time()

            # Simple shape detection using hand contours
            hand_contours = detect_hand_contours(frame)

            if hand_contours and current_time - st.session_state.last_shape_time > 3:
                # Get the largest contour (hand)
                largest_contour = max(hand_contours, key=cv2.contourArea)
                
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Determine shape based on number of vertices
                vertices = len(approx)
                if vertices < 3:
                    detected = "unknown"
                elif vertices == 3:
                    detected = "triangle"
                elif vertices == 4:
                    # Check if it's a square or rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    detected = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
                elif vertices == 5:
                    detected = "pentagon"
                elif vertices >= 6:
                    # Check if it's a circle
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    detected = "circle" if circularity > 0.7 else "unknown"
                else:
                    detected = "unknown"

                st.session_state.detected_shape = detected

                # Check if detected shape matches target
                if detected == st.session_state.target_shape:
                    st.session_state.geometry_score += 15
                    st.session_state.target_shape = random.choice(
                        ["circle", "triangle", "square", "star"])
                    st.balloons()

                st.session_state.last_shape_time = current_time

            # Draw shape text on frame
            cv2.putText(frame,
                        f"Detected: {st.session_state.detected_shape}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            # Draw contours
            cv2.drawContours(frame, hand_contours, -1, (0, 255, 0), 2)

            return frame

        while run_camera and st.session_state.camera.isOpened():
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = process_geometry_frame(frame)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    else:
        if 'camera' in st.session_state and st.session_state.camera.isOpened():
            st.session_state.camera.release()

    st.write("### How to Play:")
    st.write("- Look at the shape you need to make")
    st.write("- Form that shape with your hand clearly")
    st.write("- The AI will try to detect your hand shape")
    st.write("- Get points for correct shapes!")

    if st.button("🔄 New Shape", key="new_shape"):
        st.session_state.target_shape = random.choice(
            ["circle", "triangle", "square", "star"])
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_geometry"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Geometry Gesture",
             st.session_state.geometry_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.geometry_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Science Lab Simulator
def science_lab_game():
    st.header("🧪 Science Lab Simulator")

    st.write("""
    Perform virtual science experiments with hand gestures!
    Mix different elements and see what happens.
    """)

    # Initialize game state
    if 'science_score' not in st.session_state:
        st.session_state.science_score = 0
        st.session_state.current_experiment = random.choice([
            "Mix red and blue",
            "Add heat to water",
            "Combine acid and base"
        ])
        st.session_state.experiment_result = ""
        st.session_state.last_experiment_time = 0

    st.subheader(f"Score: {st.session_state.science_score}")
    st.markdown(
        f"<h2 style='text-align: center; color: #4CAF50;'>Experiment: {st.session_state.current_experiment}</h2>",
        unsafe_allow_html=True)

    run_camera = st.checkbox("Start Camera", key="science_cam")

    if run_camera:
        if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FRAME_WINDOW = st.image([])

        def process_science_frame(frame):
            current_time = time.time()

            # Detect hand movements for experiment actions
            hand_positions, fgMask = detect_hand_position(frame, st.session_state.backSub)

            if hand_positions and current_time - st.session_state.last_experiment_time > 5:
                # Simulate experiment based on hand movements
                if "Mix" in st.session_state.current_experiment:
                    st.session_state.experiment_result = "Created purple solution! 🟣"
                    st.session_state.science_score += 20
                
                elif "heat" in st.session_state.current_experiment.lower():
                    # Check for heating motion (hand moving upward)
                    st.session_state.experiment_result = "Water turned to steam! 💨"
                    st.session_state.science_score += 25
                
                elif "acid" in st.session_state.current_experiment.lower():
                    # Check for pouring motion
                    st.session_state.experiment_result = "Neutralization reaction! ⚗️"
                    st.session_state.science_score += 30

                # New experiment
                st.session_state.current_experiment = random.choice([
                    "Mix red and blue",
                    "Add heat to water",
                    "Combine acid and base"
                ])
                st.session_state.last_experiment_time = current_time
                st.balloons()

            # Draw experiment instructions
            cv2.putText(frame,
                        f"Perform: {st.session_state.current_experiment}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

            if st.session_state.experiment_result:
                cv2.putText(frame,
                            f"Result: {st.session_state.experiment_result}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2)

            return frame

        while run_camera and st.session_state.camera.isOpened():
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = process_science_frame(frame)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    else:
        if 'camera' in st.session_state and st.session_state.camera.isOpened():
            st.session_state.camera.release()

    st.write("### How to Play:")
    st.write("- Read the experiment instructions")
    st.write("- Perform the action with hand gestures")
    st.write("- See the scientific result")
    st.write("- Learn about different reactions!")

    if st.button("🔄 New Experiment", key="new_experiment"):
        st.session_state.current_experiment = random.choice([
            "Mix red and blue",
            "Add heat to water",
            "Combine acid and base"
        ])
        st.session_state.experiment_result = ""
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_science"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Science Lab",
             st.session_state.science_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.science_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# English Word Builder Game
def english_word_game():
    st.header("🔤 English Word Builder")

    st.write("""
    Form words with hand gestures and voice commands!
    Practice spelling and vocabulary in a fun way.
    """)

    # Initialize game state
    if 'english_score' not in st.session_state:
        st.session_state.english_score = 0
        st.session_state.current_word = random.choice([
            "cat", "dog", "sun", "moon", "star",
            "tree", "book", "ball", "house", "water"
        ])
        st.session_state.formed_letters = []
        st.session_state.last_letter_time = 0

    st.subheader(f"Score: {st.session_state.english_score}")
    st.markdown(
        f"<h2 style='text-align: center; color: #FF6B6B;'>Spell: {st.session_state.current_word.upper()}</h2>",
        unsafe_allow_html=True)

    # Display formed letters
    if st.session_state.formed_letters:
        st.write("Formed letters:")
        st.markdown(
            f"<h3 style='text-align: center; color: #4CAF50;'>{' '.join(st.session_state.formed_letters).upper()}</h3>",
            unsafe_allow_html=True)

    run_camera = st.checkbox("Start Camera", key="english_cam")

    if run_camera:
        if 'camera' not in st.session_state or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FRAME_WINDOW = st.image([])

        def process_english_frame(frame):
            current_time = time.time()

            # Simple letter detection using hand gestures
            hand_contours = detect_hand_contours(frame)

            if hand_contours and current_time - st.session_state.last_letter_time > 3:
                # Count fingers to determine letter
                largest_contour = max(hand_contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
                
                try:
                    defects = cv2.convexityDefects(largest_contour, hull_indices)
                    
                    if defects is not None:
                        # Count fingers based on convexity defects
                        finger_count = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            if d > 10000:  # Filter small defects
                                finger_count += 1
                        
                        # Map finger count to letters (A=1, B=2, etc.)
                        if 1 <= finger_count <= 5:
                            letter = chr(64 + finger_count)  # A to E
                            st.session_state.formed_letters.append(letter)
                            
                            # Check if word is complete
                            formed_word = ''.join(st.session_state.formed_letters)
                            if formed_word.lower() == st.session_state.current_word:
                                st.session_state.english_score += len(st.session_state.current_word) * 5
                                st.session_state.current_word = random.choice([
                                    "cat", "dog", "sun", "moon", "star",
                                    "tree", "book", "ball", "house", "water"
                                ])
                                st.session_state.formed_letters = []
                                st.balloons()
                            
                            st.session_state.last_letter_time = current_time
                
                except Exception as e:
                    pass

            # Draw instructions
            cv2.putText(frame,
                        "Show fingers to form letters (1-5 fingers = A-E)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            return frame

        while run_camera and st.session_state.camera.isOpened():
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = process_english_frame(frame)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    else:
        if 'camera' in st.session_state and st.session_state.camera.isOpened():
            st.session_state.camera.release()

    # Voice input for spelling
    if st.button("🎤 Say the Word", key="say_word"):
        with st.spinner("Listening..."):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=3)
                    spoken_word = recognizer.recognize_google(audio).lower()

                    if spoken_word == st.session_state.current_word:
                        st.session_state.english_score += len(st.session_state.current_word) * 10
                        st.session_state.current_word = random.choice([
                            "cat", "dog", "sun", "moon", "star",
                            "tree", "book", "ball", "house", "water"
                        ])
                        st.session_state.formed_letters = []
                        st.success("Correct! 🎉")
                        st.balloons()
                    else:
                        st.error(f"Incorrect. You said: {spoken_word}")

                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    st.error("Couldn't understand. Try again!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("🔄 New Word", key="new_word"):
        st.session_state.current_word = random.choice([
            "cat", "dog", "sun", "moon", "star",
            "tree", "book", "ball", "house", "water"
        ])
        st.session_state.formed_letters = []
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_english"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "English Word Builder",
             st.session_state.english_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.english_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Drawing Recognition Game
def drawing_recognition_game():
    st.header("🎨 Drawing Recognition")

    st.write("""
    Draw shapes and objects for the AI to recognize!
    Practice your drawing skills and see if the AI can guess what you drew.
    """)

    # Initialize game state
    if 'drawing_score' not in st.session_state:
        st.session_state.drawing_score = 0
        st.session_state.target_object = random.choice([
            "circle", "square", "triangle", "house", "tree",
            "sun", "star", "heart", "smiley", "car"
        ])
        st.session_state.detected_object = ""
        st.session_state.last_detection_time = 0

    st.subheader(f"Score: {st.session_state.drawing_score}")
    st.markdown(
        f"<h2 style='text-align: center; color: #4CAF50;'>Draw a: {st.session_state.target_object.upper()}</h2>",
        unsafe_allow_html=True)

    # Create drawing canvas
    canvas_size = 400
    if 'drawing_canvas' not in st.session_state:
        st.session_state.drawing_canvas = np.ones(
            (canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # Display canvas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(st.session_state.drawing_canvas, channels="BGR",
                 use_column_width=True, caption="Drawing Canvas")

        # Drawing controls
        if st.button("🖌️ Start Drawing", key="start_drawing"):
            # Reset canvas
            st.session_state.drawing_canvas = np.ones(
                (canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        if st.button("🔍 Analyze Drawing", key="analyze_drawing"):
            possible_objects = [
                "circle", "square", "triangle", "house", "tree",
                "sun", "star", "heart", "smiley", "car"
            ]
            st.session_state.detected_object = random.choice(possible_objects)

            # Check if detected object matches target
            if st.session_state.detected_object == st.session_state.target_object:
                st.session_state.drawing_score += 25
                st.session_state.target_object = random.choice(possible_objects)
                st.success("Correct! 🎉")
                st.balloons()
            else:
                st.error(f"AI guessed: {st.session_state.detected_object}")

    with col2:
        st.write("### Drawing Tips:")
        st.write("- Draw clearly and simply")
        st.write("- Use the whole canvas")
        st.write("- Common shapes work best")
        st.write("- The AI will try to guess what you drew")

        if st.session_state.detected_object:
            st.write(f"**AI guessed:** {st.session_state.detected_object}")

    # Instructions for actual drawing (would need a proper drawing interface)
    st.info("""
    Note: This is a simplified version. In a full implementation, 
    you would use a proper drawing interface with mouse/touch input 
    and a trained image recognition model.
    """)

    if st.button("🔄 New Object", key="new_drawing_object"):
        st.session_state.target_object = random.choice([
            "circle", "square", "triangle", "house", "tree",
            "sun", "star", "heart", "smiley", "car"
        ])
        st.session_state.detected_object = ""
        st.session_state.drawing_canvas = np.ones(
            (canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_drawing"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Drawing Recognition",
             st.session_state.drawing_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.drawing_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Music Rhythm Game
def music_rhythm_game():
    st.header("🎵 Music Rhythm Game")

    st.write("""
    Clap to the rhythm and create music patterns!
    Develop your sense of rhythm and timing.
    """)

    # Initialize game state
    if 'music_score' not in st.session_state:
        st.session_state.music_score = 0
        st.session_state.rhythm_pattern = [1, 0, 1, 0, 1, 1, 0, 1]  # 1=clap, 0=rest
        st.session_state.user_pattern = []
        st.session_state.current_beat = 0
        st.session_state.last_clap_time = 0
        st.session_state.playing = False

    st.subheader(f"Score: {st.session_state.music_score}")

    # Display rhythm pattern
    st.write("Rhythm Pattern:")
    pattern_display = "".join(["👏" if beat == 1 else "🤚" for beat in st.session_state.rhythm_pattern])
    st.markdown(f"<h3 style='text-align: center;'>{pattern_display}</h3>", unsafe_allow_html=True)

    # Game controls
    if st.button("▶️ Start Rhythm", key="start_rhythm"):
        st.session_state.playing = True
        st.session_state.user_pattern = []
        st.session_state.current_beat = 0
        st.session_state.last_clap_time = time.time()

    if st.button("👏 I Clapped!", key="clap_detected"):
        if st.session_state.playing:
            current_time = time.time()
            st.session_state.user_pattern.append(1)  # Record clap
            st.session_state.last_clap_time = current_time
            st.success("Clap detected! 👏")

    # Rhythm game logic
    if st.session_state.playing:
        current_time = time.time()
        beat_interval = 1.0  # 1 second per beat

        # Check if it's time for the next beat
        if current_time - st.session_state.last_clap_time > beat_interval:
            st.session_state.user_pattern.append(0)  # Record missed beat
            st.session_state.last_clap_time = current_time
            st.session_state.current_beat += 1

            # Check if pattern is complete
            if len(st.session_state.user_pattern) >= len(st.session_state.rhythm_pattern):
                st.session_state.playing = False
                
                # Calculate accuracy
                correct = 0
                for i in range(len(st.session_state.rhythm_pattern)):
                    if st.session_state.user_pattern[i] == st.session_state.rhythm_pattern[i]:
                        correct += 1
                
                accuracy = correct / len(st.session_state.rhythm_pattern)
                st.session_state.music_score += int(accuracy * 100)
                
                if accuracy > 0.8:
                    st.success(f"Great rhythm! Accuracy: {accuracy:.0%} 🎉")
                    st.balloons()
                else:
                    st.warning(f"Keep practicing! Accuracy: {accuracy:.0%}")

    # Display current progress
    if st.session_state.user_pattern:
        user_display = "".join(["👏" if beat == 1 else "🤚" for beat in st.session_state.user_pattern])
        st.write("Your pattern:")
        st.markdown(f"<h4 style='text-align: center;'>{user_display}</h4>", unsafe_allow_html=True)

    if st.button("🔄 New Pattern", key="new_rhythm"):
        st.session_state.rhythm_pattern = [random.choice([0, 1]) for _ in range(8)]
        st.session_state.user_pattern = []
        st.session_state.current_beat = 0
        st.session_state.playing = False
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_music"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Music Rhythm",
             st.session_state.music_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.music_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Puzzle Solver Game
def puzzle_solver_game():
    st.header("🧩 Puzzle Solver")

    st.write("""
    Solve puzzles using hand gestures and voice commands!
    Exercise your problem-solving skills with various puzzle types.
    """)

    # Initialize game state
    if 'puzzle_score' not in st.session_state:
        st.session_state.puzzle_score = 0
        st.session_state.current_puzzle = None
        st.session_state.puzzle_type = random.choice(["memory", "pattern", "logic"])
        st.session_state.generate_puzzle()

    # Puzzle generation
    def generate_puzzle(puzzle_type):
        if puzzle_type == "memory":
            # Memory sequence
            sequence = [random.randint(1, 4) for _ in range(4)]
            return {
                "type": "memory",
                "question": "Remember this sequence:",
                "data": sequence,
                "answer": sequence
            }
        
        elif puzzle_type == "pattern":
            # Pattern recognition
            patterns = [
                ["🔴", "🔵", "🔴", "🔵", "🔴"],  # Red, Blue alternating
                ["⭐", "⭐", "🌙", "⭐", "⭐"],     # Star, Star, Moon, Star, Star
                ["▲", "▼", "▲", "▼", "▲"]        # Up, Down alternating
            ]
            pattern = random.choice(patterns)
            return {
                "type": "pattern",
                "question": "What comes next in this pattern?",
                "data": pattern,
                "answer": pattern[0]  # Next should be the first element
            }
        
        else:  # logic
            # Simple logic puzzle
            puzzles = [
                {
                    "question": "If all apples are fruits, and this is an apple, then what is it?",
                    "data": "🍎",
                    "answer": "fruit"
                },
                {
                    "question": "What is 2 + 3?",
                    "data": "5",
                    "answer": "5"
                }
            ]
            return random.choice(puzzles)

    # Generate puzzle if needed
    if st.session_state.current_puzzle is None:
        st.session_state.current_puzzle = generate_puzzle(st.session_state.puzzle_type)

    st.subheader(f"Score: {st.session_state.puzzle_score}")

    # Display current puzzle
    puzzle = st.session_state.current_puzzle
    st.markdown(f"<h3 style='color: #4CAF50;'>{puzzle['question']}</h3>", unsafe_allow_html=True)
    
    if puzzle["type"] == "memory":
        st.write(" ".join([str(x) for x in puzzle["data"]]))
        time.sleep(2)  # Give time to memorize
        st.write("Sequence hidden... What was the sequence?")
    
    elif puzzle["type"] == "pattern":
        st.write(" ".join(puzzle["data"]))
        st.write("What comes next?")
    
    else:  # logic
        st.write(puzzle["data"])

    # Answer input
    user_answer = st.text_input("Your Answer:", key="puzzle_answer")
    
    if st.button("Submit Answer", key="submit_puzzle"):
        if puzzle["type"] == "memory":
            # Check if sequence matches
            try:
                user_sequence = [int(x) for x in user_answer.split()]
                if user_sequence == puzzle["answer"]:
                    st.session_state.puzzle_score += 30
                    st.success("Correct sequence! 🎉")
                    st.balloons()
                else:
                    st.error("Incorrect sequence. Try again!")
            except ValueError:
                st.error("Please enter numbers separated by spaces")
        
        elif puzzle["type"] == "pattern":
            if user_answer.strip() == puzzle["answer"]:
                st.session_state.puzzle_score += 25
                st.success("Correct pattern! 🎉")
                st.balloons()
            else:
                st.error("Incorrect pattern. Try again!")
        
        else:  # logic
            if user_answer.lower().strip() == puzzle["answer"].lower():
                st.session_state.puzzle_score += 20
                st.success("Correct answer! 🎉")
                st.balloons()
            else:
                st.error("Incorrect answer. Try again!")

        # New puzzle
        st.session_state.puzzle_type = random.choice(["memory", "pattern", "logic"])
        st.session_state.current_puzzle = generate_puzzle(st.session_state.puzzle_type)
        st.experimental_rerun()

    # Voice answer option
    if st.button("🎤 Answer with Voice", key="voice_puzzle"):
        with st.spinner("Listening..."):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5)
                    spoken_answer = recognizer.recognize_google(audio)
                    
                    # Check answer based on puzzle type
                    if puzzle["type"] == "memory":
                        try:
                            spoken_numbers = [int(x) for x in spoken_answer.split() if x.isdigit()]
                            if spoken_numbers == puzzle["answer"]:
                                st.session_state.puzzle_score += 30
                                st.success("Correct sequence! 🎉")
                                st.balloons()
                            else:
                                st.error("Incorrect sequence. Try again!")
                        except ValueError:
                            st.error("Couldn't understand numbers")
                    
                    elif puzzle["type"] == "pattern":
                        if spoken_answer.strip() == puzzle["answer"]:
                            st.session_state.puzzle_score += 25
                            st.success("Correct pattern! 🎉")
                            st.balloons()
                        else:
                            st.error("Incorrect pattern. Try again!")
                    
                    else:  # logic
                        if spoken_answer.lower().strip() == puzzle["answer"].lower():
                            st.session_state.puzzle_score += 20
                            st.success("Correct answer! 🎉")
                            st.balloons()
                        else:
                            st.error("Incorrect answer. Try again!")

                    # New puzzle
                    st.session_state.puzzle_type = random.choice(["memory", "pattern", "logic"])
                    st.session_state.current_puzzle = generate_puzzle(st.session_state.puzzle_type)
                    st.experimental_rerun()

                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    st.error("Couldn't understand. Try again!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("🔄 New Puzzle", key="new_puzzle"):
        st.session_state.puzzle_type = random.choice(["memory", "pattern", "logic"])
        st.session_state.current_puzzle = generate_puzzle(st.session_state.puzzle_type)
        st.experimental_rerun()

    if st.button("💾 Save Score", key="save_puzzle"):
        conn = sqlite3.connect('education_platform.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO game_scores (user_id, game_name, score) VALUES (?, ?, ?)",
            (st.session_state.user['id'],
             "Puzzle Solver",
             st.session_state.puzzle_score))
        c.execute(
            "UPDATE users SET total_score = total_score + ? WHERE id = ?",
            (st.session_state.puzzle_score,
             st.session_state.user['id']))
        conn.commit()
        conn.close()
        st.success("Score saved successfully!")

# Run the main application
if __name__ == "__main__":
    main()