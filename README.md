# 🥽 Real-Time VR Live Captions, Diarized Translation & Post-Meeting MoM AI

> **Speech & Language Processing (SLP) Course Project**  
> Real-Time Multilingual Live Captions in Virtual Reality + Speaker Diarization + Post-Meeting Minutes of Meeting (MoM) Intelligence Generator.

---

## 🌟 Key Features

1. **Feature 1: Real-Time VR Live Captions & Diarized Translation**
   - **Ultra-Low Latency Streaming**: Speech-to-text with $<240\text{ms}$ latency powered by GPU-accelerated `faster-whisper`.
   - **Live Acoustic Speaker Diarization**: Color-coded speaker tags (`Speaker 1`, `Speaker 2`, etc.) based on acoustic spectral feature vectors.
   - **Multilingual Translation**: Live target translation into English, Spanish, French, German, Hindi, Tamil, Chinese, Japanese, etc.
   - **3D Spatial VR HUD Simulator**: Glassmorphic Heads-Up Display simulating Meta Quest 3 & Apple Vision Pro HMD overlays with spatial audio spectrum visualizers.

2. **Feature 2: Virtual Meeting Assistant & Post-Meeting MoM Studio**
   - **Live & File Input Support**: Stream live laptop microphone speech or upload pre-recorded audio files (`.wav`, `.mp3`).
   - **Minutes of Meeting (MoM) Intelligence Generator**:
     - Abstractive Neural Meeting Summary (BART / NLP TextRank).
     - Extracted Action Items Matrix with assigned speakers, deliverables, and deadlines.
     - Speaker Talk-Time Distribution & Interaction Metrics.
     - Interactive Speaker Diarization Timeline (Gantt Chart).
   - **Multi-Format Reports**: Export downloadable HTML reports, Markdown (`.md`), and JSON.

---

## 🏗️ Architecture Pipeline

```
[ Microphone / Audio File Stream ]
            │
            ▼
[ Voice Activity Detection & Chunking ] ──► [ Audio Processing Queue ]
                                                  │
            ┌─────────────────────────────────────┴─────────────────────────────────────┐
            ▼                                                                           ▼
[ Faster-Whisper ASR (CUDA GPU) ]                                     [ Acoustic Speaker Diarization ]
            │                                                                           │
            ▼                                                                           ▼
[ Detected Speech & Source Lang ] ──► [ Neural Machine Translation ] ──► [ Assigned Speaker Tag ]
                                                    │
                                                    ▼
                                     [ Real-Time SSE Subtitle Stream ]
                                                    │
                                                    ├──► [ Feature 1: VR HUD View ]
                                                    └──► [ Feature 2: Meeting & MoM AI ]
```

---

## 🚀 Installation & Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/vr-captions-mom-ai.git
cd vr-captions-mom-ai
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
python app.py
```

### 3. Open Web Interface
Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 📁 Repository Structure

```
├── app.py                             # Flask Web Server & SSE Streaming API
├── slp_engine.py                      # VAD, ASR, Translation & Diarization Engine
├── mom_generator.py                   # Post-Meeting Summary & Action Items Generator
├── realtime_transcribe_diarize_translate.py  # Reference CLI implementation
├── generate_test_audio.py             # Utility to generate test audio files
├── sample_meeting.wav                 # Sample meeting audio for testing
├── SLP_Project_Report.md              # Academic SLP Project Report
├── vr_integration.md                  # Unity VR C# WebSocket Client Integration Blueprint
├── requirements.txt                   # Python dependencies
├── templates/
│   └── index.html                     # Web App Single-Page Interface
└── static/
    ├── css/style.css                  # Dark mode glassmorphic styling & 3D VR effects
    └── js/main.js                     # SSE listener & Canvas audio spectrum visualizer
```

---

## 📄 Documentation & Reports
- [Academic Project Report](SLP_Project_Report.md)
- [Unity VR Integration Guide](vr_integration.md)
