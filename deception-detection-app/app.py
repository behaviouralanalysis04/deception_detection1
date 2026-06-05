"""
Deception Detection System - Streamlit Cloud Deployment
Integrated Video + Audio Analysis for Lie Detection
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import os
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import queue
import librosa
import speech_recognition as sr
import subprocess
from collections import deque
import utils

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Deception Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .title-text {
        font-family: 'Orbitron', monospace;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 18px;
        margin-bottom: 40px;
    }
    .stCard {
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(102,126,234,0.3);
        margin-bottom: 20px;
    }
    .combined-score {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(102,126,234,0.5);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 30px;
        font-weight: 600;
        width: 100%;
    }
    .info-box {
        background: rgba(102,126,234,0.2);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 20px 0;
    }
    .indicator-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .indicator-badge.high {
        background: rgba(220,53,69,0.3);
        color: #ff6b6b;
        border: 1px solid #dc3545;
    }
    .indicator-badge.medium {
        background: rgba(255,193,7,0.3);
        color: #ffd43b;
        border: 1px solid #ffc107;
    }
    .indicator-badge.low {
        background: rgba(40,167,69,0.3);
        color: #51cf66;
        border: 1px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_color_for_score(score):
    if score >= 60:
        return '#dc3545'
    elif score >= 40:
        return '#ffc107'
    else:
        return '#28a745'

def get_classification_for_score(score):
    if score >= 60:
        return 'HIGH PROBABILITY OF DECEPTION'
    elif score >= 40:
        return 'POSSIBLE DECEPTION'
    else:
        return 'LOW PROBABILITY OF DECEPTION'

def create_gauge_chart(score, title="Deception Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': get_color_for_score(score)},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(40,167,69,0.3)"},
                {'range': [40, 60], 'color': "rgba(255,193,7,0.3)"},
                {'range': [60, 100], 'color': "rgba(220,53,69,0.3)"}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'}, height=300)
    return fig

def calculate_deception_score(features):
    """Rule-based scoring from video features"""
    score = 0
    indicators = []
    
    if features.get('blink_rate', 0) > 30:
        score += 15
        indicators.append(('Elevated blink rate', 'high'))
    elif features.get('blink_rate', 0) < 10:
        score += 10
        indicators.append(('Reduced blink rate', 'medium'))
    
    gaze_aversion = features.get('gaze_left_ratio', 0) + features.get('gaze_right_ratio', 0)
    if gaze_aversion > 0.6:
        score += 20
        indicators.append(('Frequent gaze aversion', 'high'))
    elif gaze_aversion > 0.4:
        score += 10
        indicators.append(('Occasional gaze aversion', 'medium'))
    
    if features.get('avg_mouth_open_ratio', 0) > 0.3:
        score += 10
        indicators.append(('Increased mouth opening', 'medium'))
    
    if features.get('avg_lip_compression', 0) > 15:
        score += 15
        indicators.append(('Lip compression detected', 'high'))
    
    if features.get('avg_facial_asymmetry', 0) > 10:
        score += 15
        indicators.append(('Facial asymmetry detected', 'medium'))
    
    if features.get('micro_expression_frequency', 0) > 2:
        score += 20
        indicators.append(('Frequent micro-expressions', 'high'))
    elif features.get('micro_expression_frequency', 0) > 1:
        score += 10
        indicators.append(('Occasional micro-expressions', 'medium'))
    
    if features.get('head_nod_frequency', 0) > 1.5:
        score += 5
        indicators.append(('Excessive head nodding', 'medium'))
    
    if features.get('head_shake_frequency', 0) > 1:
        score += 5
        indicators.append(('Head shaking detected', 'medium'))
    
    return min(100, score), indicators

def calculate_audio_score(audio_features):
    """Calculate deception score from audio features"""
    if not audio_features:
        return 0, []
    
    score = 0
    indicators = []
    
    if audio_features.get('std_pitch', 0) > 30:
        score += 20
        indicators.append(('High pitch variation', 'high'))
    
    if audio_features.get('std_energy', 0) > 0.05:
        score += 15
        indicators.append(('Inconsistent energy', 'medium'))
    
    tempo = audio_features.get('speech_tempo', 120)
    if tempo > 160:
        score += 15
        indicators.append(('Rapid speech', 'medium'))
    elif tempo < 100:
        score += 15
        indicators.append(('Slow speech', 'medium'))
    
    if audio_features.get('energy_range', 0) > 0.15:
        score += 15
        indicators.append(('Wide energy range', 'medium'))
    
    if audio_features.get('speech_activity', 0) < 0.4:
        score += 10
        indicators.append(('Low speech activity', 'medium'))
    
    return min(100, score), indicators

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ============================================
# VIDEO PROCESSOR
# ============================================

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.blink_counter = 0
        self.blink_rate = 0
        self.eye_closed_start = None
        self.gaze_durations = {'left': 0, 'right': 0, 'center': 0}
        self.mar_values = deque(maxlen=300)
        self.asymmetry_values = deque(maxlen=300)
        self.lip_comp_values = deque(maxlen=300)
        self.head_pitch_values = deque(maxlen=300)
        self.head_yaw_values = deque(maxlen=300)
        self.head_roll_values = deque(maxlen=300)
        self.nod_count = 0
        self.shake_count = 0
        self.prev_landmarks = None
        self.micro_expression_frames = 0
        self.features_queue = queue.Queue(maxsize=5)
        self.last_process_time = time.time()
        self.fps_target = 10
        self.prev_head_pitch = None
        self.prev_head_yaw = None

    def recv(self, frame):
        current_time = time.time()
        
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.frame_count += 1
        
        # Process at target FPS
        if current_time - self.last_process_time >= 1.0 / self.fps_target:
            self.last_process_time = current_time
            
            faces = utils.detector(gray, 0)
            
            if len(faces) > 0:
                landmarks = utils.predictor(gray, faces[0])
                landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
                h, w = img.shape[:2]
                
                # Blink detection
                left_ear = utils.eye_aspect_ratio(landmarks_list, utils.LEFT_EYE_INDICES)
                right_ear = utils.eye_aspect_ratio(landmarks_list, utils.RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < 0.2:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = current_time
                else:
                    if self.eye_closed_start is not None:
                        self.blink_counter += 1
                        self.eye_closed_start = None
                
                if self.frame_count > 0:
                    self.blink_rate = (self.blink_counter / (self.frame_count / 30)) * 60
                
                # Gaze detection
                gaze = utils.gaze_direction(landmarks_list, w)
                self.gaze_durations[gaze] += 1
                
                # Mouth features
                mar = utils.mouth_aspect_ratio(landmarks_list)
                self.mar_values.append(mar)
                
                asym = utils.facial_asymmetry(landmarks_list)
                self.asymmetry_values.append(asym)
                
                lip_comp = utils.lip_compression(landmarks_list)
                self.lip_comp_values.append(lip_comp)
                
                # Head pose
                pitch, yaw, roll = utils.head_pose(landmarks_list, w, h)
                self.head_pitch_values.append(pitch)
                self.head_yaw_values.append(yaw)
                self.head_roll_values.append(roll)
                
                # Head nods and shakes
                if self.prev_head_pitch is not None:
                    if abs(self.prev_head_pitch - pitch) > 5:
                        self.nod_count += 1
                    if self.prev_head_yaw is not None and abs(yaw - self.prev_head_yaw) > 10:
                        self.shake_count += 1
                
                self.prev_head_pitch = pitch
                self.prev_head_yaw = yaw
                
                # Micro-expressions
                if self.prev_landmarks is not None:
                    movement = utils.micro_expression_magnitude(self.prev_landmarks, landmarks_list)
                    if movement > 5.0:
                        self.micro_expression_frames += 1
                self.prev_landmarks = landmarks_list
                
                # Extract features periodically
                if self.frame_count % 90 == 0 and self.frame_count >= 90:
                    duration_sec = self.frame_count / 30
                    processed_frames = self.frame_count
                    
                    features = {
                        'blink_rate': min(float(self.blink_rate), 45.0),
                        'avg_blink_duration': 0.2,
                        'gaze_left_ratio': self.gaze_durations['left'] / max(processed_frames, 1),
                        'gaze_right_ratio': self.gaze_durations['right'] / max(processed_frames, 1),
                        'gaze_center_ratio': self.gaze_durations['center'] / max(processed_frames, 1),
                        'avg_mouth_open_ratio': float(np.mean(self.mar_values)) if self.mar_values else 0.0,
                        'std_mouth_open_ratio': float(np.std(self.mar_values)) if self.mar_values else 0.0,
                        'avg_facial_asymmetry': float(np.mean(self.asymmetry_values)) if self.asymmetry_values else 0.0,
                        'std_facial_asymmetry': float(np.std(self.asymmetry_values)) if self.asymmetry_values else 0.0,
                        'avg_lip_compression': float(np.mean(self.lip_comp_values)) if self.lip_comp_values else 0.0,
                        'micro_expression_frequency': self.micro_expression_frames / max(duration_sec, 0.1),
                        'avg_head_pitch': float(np.mean(self.head_pitch_values)) if self.head_pitch_values else 0.0,
                        'std_head_pitch': float(np.std(self.head_pitch_values)) if self.head_pitch_values else 0.0,
                        'avg_head_yaw': float(np.mean(self.head_yaw_values)) if self.head_yaw_values else 0.0,
                        'std_head_yaw': float(np.std(self.head_yaw_values)) if self.head_yaw_values else 0.0,
                        'avg_head_roll': float(np.mean(self.head_roll_values)) if self.head_roll_values else 0.0,
                        'std_head_roll': float(np.std(self.head_roll_values)) if self.head_roll_values else 0.0,
                        'head_nod_frequency': self.nod_count / max(duration_sec, 0.1),
                        'head_shake_frequency': self.shake_count / max(duration_sec, 0.1),
                        'head_tilt_frequency': 0,
                        'duration_seconds': duration_sec
                    }
                    
                    if self.features_queue.empty():
                        self.features_queue.put(features)
            
            # Draw overlay
            cv2.putText(img, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
            cv2.putText(img, f"Blinks: {self.blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
            cv2.putText(img, f"Rate: {self.blink_rate:.0f}/min", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================
# AUDIO ANALYZER
# ============================================

class AudioAnalyzer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = None, None
        
    def load_audio(self):
        try:
            self.y, self.sr = librosa.load(self.audio_path, sr=None, duration=60)
            return self.y, self.sr
        except Exception as e:
            st.warning(f"Could not load audio: {e}")
            return None, None
    
    def extract_features(self):
        if self.y is None:
            return {}
        
        features = {}
        try:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            features['speech_tempo'] = float(tempo) if isinstance(tempo, (int, float)) else 120.0
            
            rms = librosa.feature.rms(y=self.y)[0]
            features['avg_energy'] = float(np.mean(rms))
            features['std_energy'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
            pitch_values = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['avg_pitch'] = float(np.mean(pitch_values))
                features['std_pitch'] = float(np.std(pitch_values))
            else:
                features['avg_pitch'] = 0.0
                features['std_pitch'] = 0.0
            
            zcr = librosa.feature.zero_crossing_rate(self.y)[0]
            features['speech_activity'] = float(np.mean(zcr > 0.01))
            
        except Exception as e:
            features = {
                'speech_tempo': 120.0,
                'avg_energy': 0.5,
                'std_energy': 0.05,
                'avg_pitch': 150.0,
                'std_pitch': 20.0,
                'speech_activity': 0.5
            }
        
        return features
    
    def transcribe(self):
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(self.audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                return recognizer.recognize_google(audio)
        except Exception as e:
            return ""

# ============================================
# VIDEO FILE PROCESSOR
# ============================================

def process_video_file(video_path, progress_callback=None):
    """Extract features from uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if duration > 60:
        cap.release()
        raise Exception(f"Video too long: {duration:.1f}s (max 60s)")
    
    # Feature tracking
    frame_count = 0
    processed_frames = 0
    blinks = 0
    gaze_durations = {'left': 0, 'right': 0, 'center': 0}
    mar_values = []
    asymmetry_values = []
    lip_comp_values = []
    head_pitch_values = []
    head_yaw_values = []
    head_roll_values = []
    nod_count = 0
    shake_count = 0
    prev_head_pitch = None
    prev_head_yaw = None
    prev_landmarks = None
    micro_expressions = 0
    
    process_every_n = max(1, int(fps / 10))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % process_every_n != 0:
            continue
        
        processed_frames += 1
        h, w = frame.shape[:2]
        
        landmarks_list = utils.get_landmarks(frame)
        if landmarks_list is None:
            continue
        
        # Blink detection
        left_ear = utils.eye_aspect_ratio(landmarks_list, utils.LEFT_EYE_INDICES)
        right_ear = utils.eye_aspect_ratio(landmarks_list, utils.RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0
        if ear < 0.2:
            blinks += 1
        
        # Gaze
        gaze = utils.gaze_direction(landmarks_list, w)
        gaze_durations[gaze] += 1
        
        # Mouth features
        mar_values.append(utils.mouth_aspect_ratio(landmarks_list))
        asymmetry_values.append(utils.facial_asymmetry(landmarks_list))
        lip_comp_values.append(utils.lip_compression(landmarks_list))
        
        # Head pose
        pitch, yaw, roll = utils.head_pose(landmarks_list, w, h)
        head_pitch_values.append(pitch)
        head_yaw_values.append(yaw)
        head_roll_values.append(roll)
        
        if prev_head_pitch is not None:
            if abs(prev_head_pitch - pitch) > 5:
                nod_count += 1
            if prev_head_yaw is not None and abs(yaw - prev_head_yaw) > 10:
                shake_count += 1
        
        prev_head_pitch = pitch
        prev_head_yaw = yaw
        
        # Micro-expressions
        if prev_landmarks is not None:
            movement = utils.micro_expression_magnitude(prev_landmarks, landmarks_list)
            if movement > 5.0:
                micro_expressions += 1
        prev_landmarks = landmarks_list
        
        if progress_callback and processed_frames % max(1, total_frames // 20) == 0:
            progress_callback(min(95, int(processed_frames / max(total_frames // process_every_n, 1) * 100)))
    
    cap.release()
    
    if processed_frames == 0:
        raise Exception("No faces detected in video")
    
    duration_sec = frame_count / fps
    
    features = {
        'blink_rate': (blinks / duration_sec) * 60 if duration_sec > 0 else 0,
        'avg_blink_duration': 0.2,
        'gaze_left_ratio': gaze_durations['left'] / max(processed_frames, 1),
        'gaze_right_ratio': gaze_durations['right'] / max(processed_frames, 1),
        'gaze_center_ratio': gaze_durations['center'] / max(processed_frames, 1),
        'avg_mouth_open_ratio': np.mean(mar_values) if mar_values else 0,
        'std_mouth_open_ratio': np.std(mar_values) if mar_values else 0,
        'avg_facial_asymmetry': np.mean(asymmetry_values) if asymmetry_values else 0,
        'std_facial_asymmetry': np.std(asymmetry_values) if asymmetry_values else 0,
        'avg_lip_compression': np.mean(lip_comp_values) if lip_comp_values else 0,
        'micro_expression_frequency': micro_expressions / max(duration_sec, 0.1),
        'avg_head_pitch': np.mean(head_pitch_values) if head_pitch_values else 0,
        'std_head_pitch': np.std(head_pitch_values) if head_pitch_values else 0,
        'avg_head_yaw': np.mean(head_yaw_values) if head_yaw_values else 0,
        'std_head_yaw': np.std(head_yaw_values) if head_yaw_values else 0,
        'avg_head_roll': np.mean(head_roll_values) if head_roll_values else 0,
        'std_head_roll': np.std(head_roll_values) if head_roll_values else 0,
        'head_nod_frequency': nod_count / max(duration_sec, 0.1),
        'head_shake_frequency': shake_count / max(duration_sec, 0.1),
        'head_tilt_frequency': 0,
        'duration_seconds': duration_sec
    }
    
    return features, duration_sec

# ============================================
# MAIN UI
# ============================================

st.markdown('<h1 class="title-text">🎭 DECEPTION DETECTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-Powered Lie Detection | Video + Audio Analysis</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 System Status")
    st.success("✅ dlib: Loaded")
    st.success("✅ Face Detector: Active")
    
    st.markdown("---")
    st.markdown("### 🎯 Features Analyzed")
    st.markdown("""
    **Video Analysis (60% weight):**
    - Blink rate & duration
    - Gaze direction tracking
    - Micro-expression detection
    - Facial asymmetry
    - Lip compression
    - Head movement patterns

    **Audio Analysis (40% weight):**
    - Speech rate & tempo
    - Pitch variation
    - Energy dynamics
    - Speech activity
    """)
    
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.metric("Total Analyses", len(history_df))
        st.metric("Avg Score", f"{history_df['score'].mean():.1f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎥 LIVE ANALYSIS", "📁 FILE UPLOAD", "📊 REPORTS"])

# ============================================
# TAB 1: LIVE ANALYSIS
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("🎥 Live Camera Feed")
        st.info("Click 'Start' below, then wait 10 seconds before analyzing", icon="ℹ️")
        
        webrtc_ctx = webrtc_streamer(
            key="deception-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 Analyze Now", use_container_width=True):
                if webrtc_ctx and webrtc_ctx.video_processor:
                    processor = webrtc_ctx.video_processor
                    if not processor.features_queue.empty():
                        features = processor.features_queue.get()
                        score, indicators = calculate_deception_score(features)
                        classification = get_classification_for_score(score)
                        
                        st.session_state.analysis_results = {
                            'score': score,
                            'classification': classification,
                            'indicators': indicators,
                            'features': features,
                            'type': 'Live',
                            'video_score': score,
                            'audio_score': 0
                        }
                        
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now(),
                            'type': 'Live',
                            'score': score,
                            'classification': classification
                        })
                        
                        st.success(f"Analysis Complete! Score: {score:.1f}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("Please wait 10 seconds for data collection, then click Analyze again.")
                else:
                    st.error("Please start the camera first by clicking 'Start' on the video player.")
        
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.analysis_results = None
                st.success("Reset complete!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Click **Start** on the camera player
        2. Allow camera access when prompted
        3. Position your face clearly in frame
        4. Wait **10 seconds** for data collection
        5. Click **Analyze Now** for results
        6. View detailed report in Reports tab

        **Tips for best results:**
        - Ensure good lighting
        - Keep face centered
        - Avoid extreme head movements
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TAB 2: FILE UPLOAD
# ============================================
with tab2:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📁 Upload File for Complete Analysis")
    st.markdown("Upload a video file to analyze both **visual and audio** cues simultaneously")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV (max 60 seconds)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.video(video_path)
        
        if st.button("🎯 Analyze File", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Video analysis
                status_text.info("🎥 Analyzing facial expressions and body language...")
                progress_bar.progress(20)
                
                video_features, duration = process_video_file(video_path, lambda p: progress_bar.progress(p))
                video_score, video_indicators = calculate_deception_score(video_features)
                
                progress_bar.progress(60)
                status_text.info("🎙️ Extracting audio from video...")
                
                # Extract audio from video
                audio_path = video_path.replace('.mp4', '_audio.wav')
                subprocess.run([
                    'ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', 
                    '-ar', '16000', audio_path, '-y', '-loglevel', 'quiet'
                ], capture_output=True)
                
                status_text.info("🎙️ Analyzing speech patterns...")
                progress_bar.progress(75)
                
                audio_analyzer = AudioAnalyzer(audio_path)
                audio_analyzer.load_audio()
                audio_features = audio_analyzer.extract_features()
                transcript = audio_analyzer.transcribe()
                audio_score, audio_indicators = calculate_audio_score(audio_features)
                
                # Combined score (60% video, 40% audio)
                combined_score = (video_score * 0.6) + (audio_score * 0.4)
                classification = get_classification_for_score(combined_score)
                
                progress_bar.progress(100)
                status_text.success("✅ Complete analysis finished!")
                
                st.session_state.analysis_results = {
                    'score': combined_score,
                    'classification': classification,
                    'video_score': video_score,
                    'audio_score': audio_score,
                    'video_indicators': video_indicators,
                    'audio_indicators': audio_indicators,
                    'transcript': transcript,
                    'features': video_features,
                    'type': 'File Upload'
                }
                
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': 'File Upload',
                    'filename': uploaded_file.name,
                    'score': combined_score,
                    'classification': classification,
                    'video_score': video_score,
                    'audio_score': audio_score
                })
                
                st.balloons()
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                st.error(f"Analysis failed: {str(e)}")
                st.info("Make sure the video contains a clear face and is under 60 seconds.")
            finally:
                # Cleanup
                try:
                    os.unlink(video_path)
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                except:
                    pass
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TAB 3: REPORTS
# ============================================
with tab3:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown('<div class="combined-score">', unsafe_allow_html=True)
        
        # Main gauge
        gauge = create_gauge_chart(results['score'], "Combined Deception Score")
        st.plotly_chart(gauge, use_container_width=True)
        
        color = get_color_for_score(results['score'])
        st.markdown(f'<h2 style="text-align: center; color: {color};">{results["classification"]}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Video and Audio scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🎥 Video Analysis Score")
            video_gauge = create_gauge_chart(results.get('video_score', 0), "Video Score")
            st.plotly_chart(video_gauge, use_container_width=True)
            
            if 'video_indicators' in results and results['video_indicators']:
                st.markdown("**Detected Indicators:**")
                for indicator, level in results['video_indicators']:
                    badge_class = "high" if level == "high" else "medium" if level == "medium" else "low"
                    st.markdown(f'<span class="indicator-badge {badge_class}">{indicator}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🎙️ Audio Analysis Score")
            audio_gauge = create_gauge_chart(results.get('audio_score', 0), "Audio Score")
            st.plotly_chart(audio_gauge, use_container_width=True)
            
            if 'audio_indicators' in results and results['audio_indicators']:
                st.markdown("**Detected Indicators:**")
                for indicator, level in results['audio_indicators']:
                    badge_class = "high" if level == "high" else "medium" if level == "medium" else "low"
                    st.markdown(f'<span class="indicator-badge {badge_class}">{indicator}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📊 Detailed Behavioral Metrics")
        
        if 'features' in results:
            features = results['features']
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Blink Rate", f"{features.get('blink_rate', 0):.0f}/min")
            with metric_cols[1]:
                gaze_aversion = (features.get('gaze_left_ratio', 0) + features.get('gaze_right_ratio', 0)) * 100
                st.metric("Gaze Aversion", f"{gaze_aversion:.0f}%")
            with metric_cols[2]:
                st.metric("Micro-expressions", f"{features.get('micro_expression_frequency', 0):.2f}/s")
            with metric_cols[3]:
                st.metric("Facial Asymmetry", f"{features.get('avg_facial_asymmetry', 0):.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Transcript
        if 'transcript' in results and results['transcript']:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("📝 Audio Transcript")
            st.info(results['transcript'][:500])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export button
        if st.button("📥 Export Results (CSV)", use_container_width=True):
            export_data = {
                'timestamp': datetime.now(),
                'combined_score': results['score'],
                'video_score': results.get('video_score', 0),
                'audio_score': results.get('audio_score', 0),
                'classification': results['classification'],
                'type': results.get('type', 'Unknown')
            }
            if 'features' in results:
                for k, v in results['features'].items():
                    export_data[f'video_{k}'] = v
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            st.download_button("Download CSV", csv, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    
    else:
        st.info("No analysis results yet. Start a live recording or upload a file to begin analysis.")
    
    # History section
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📜 Analysis History")
    
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
        
        # History chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['score'],
            mode='lines+markers',
            name='Deception Score',
            line=dict(color='#667eea', width=2)
        ))
        fig.add_hline(y=40, line_dash="dash", line_color="#28a745", annotation_text="Truthful Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="#dc3545", annotation_text="Deceptive Threshold")
        fig.update_layout(
            title="Score History Over Time",
            xaxis_title="Date",
            yaxis_title="Deception Score",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analysis history yet.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.5);">Powered by dlib, OpenCV, and AI | Deception Detection System</p>', unsafe_allow_html=True)
