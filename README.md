# Real-Time Multilingual Meeting Assistant

This repository contains our Speech and Language Processing course project. The application processes live microphone input or uploaded audio, generates captions, identifies speakers, translates recognised speech, and prepares a minutes-of-meeting report.

The browser interface presents the captions in a VR-style heads-up display. A separate integration note explains how the same output can be connected to a Unity-based VR client.

## Main components

- `app.py` runs the Flask web application and its server-sent event endpoints.
- `slp_engine.py` handles audio chunking, speech recognition, language detection, translation, and speaker assignment.
- `mom_generator.py` produces the meeting summary, action items, speaker statistics, and export files.
- `realtime_transcribe_diarize_translate.py` contains the standalone transcription and diarization workflow.
- `templates/index.html`, `static/css/style.css`, and `static/js/main.js` implement the browser interface.

The project accepts microphone input and uploaded audio files. Processed captions are sent to the browser as they become available. At the end of a session, the recorded transcript can be used to generate a structured meeting report.

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/AbhirajBhatta/SLP_Project.git
cd SLP_Project
pip install -r requirements.txt
```

Speaker diarization may require access to the corresponding Hugging Face model. Authenticate with the Hugging Face CLI or set the `HF_TOKEN` environment variable before running the standalone diarization script.

Start the web application with:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

## Repository contents

```text
SLP_Project/
|-- app.py
|-- slp_engine.py
|-- mom_generator.py
|-- realtime_transcribe_diarize_translate.py
|-- generate_test_audio.py
|-- sample_meeting.wav
|-- requirements.txt
|-- templates/
|-- static/
|-- projectSubmissions/
|-- SLP_Project_Report.md
`-- vr_integration.md
```

The `projectSubmissions` directory contains the Lab 4–7 LaTeX submissions, the project-report guidelines, and the final Word report.

## Documentation

- [Project report](SLP_Project_Report.md)
- [Final Word report](projectSubmissions/SLP_Final_Project_Report.docx)
- [VR integration notes](vr_integration.md)
