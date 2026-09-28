"""
app.py
Flask Application & WebServer for Real-Time VR Subtitles, Diarization & MoM Generator
"""

import os
import time
import json
import queue
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename

# Import custom SLP modules
from slp_engine import SLPSessionManager, TARGET_LANGUAGES, get_audio_input_devices
from mom_generator import MoMGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disable Flask template caching for instant UI updates
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global Session Manager & Engines
session_manager = SLPSessionManager(model_size="medium", target_lang="en")
mom_generator = MoMGenerator()

@app.after_request
def add_header(response):
    """Disable browser caching so frontend always reflects latest UI changes."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    """Render main web interface."""
    devices = get_audio_input_devices()
    return render_template('index.html', target_languages=TARGET_LANGUAGES, audio_devices=devices, cache_id=int(time.time()))

@app.route('/api/audio_devices', methods=['GET'])
def list_audio_devices():
    """API endpoint to get active microphone list."""
    return jsonify({"devices": get_audio_input_devices()})

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start real-time live captions session."""
    data = request.get_json() or {}
    audio_file_path = data.get("audio_file_path")
    target_lang = data.get("target_lang", "en")
    device_index = data.get("device_index")

    if device_index is not None:
        try:
            device_index = int(device_index)
        except ValueError:
            device_index = None

    session_manager.set_target_language(target_lang)
    session_manager.start_session(audio_file_path=audio_file_path, device_index=device_index)

    return jsonify({
        "status": "success",
        "message": "Session started",
        "session_id": session_manager.session_id,
        "target_lang": target_lang
    })

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    """Stop session and generate post-meeting Minutes of Meeting (MoM)."""
    session_id = session_manager.stop_session()
    transcript_history = list(session_manager.transcript_history)
    
    duration = 0.0
    if transcript_history:
        duration = round(transcript_history[-1].get("rel_end", 0.0), 2)

    print(f"[App] Generating MoM for session {session_id}...")
    mom_data = mom_generator.generate_mom(transcript_history, session_id, audio_duration_sec=duration)

    session_dir = session_manager.output_dir
    html_path = os.path.join(session_dir, "meeting_mom.html")
    md_path = os.path.join(session_dir, "meeting_mom.md")
    json_path = os.path.join(session_dir, "meeting_mom.json")

    mom_generator.export_html_report(mom_data, html_path)
    mom_generator.export_markdown_report(mom_data, md_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mom_data, f, indent=2, ensure_ascii=False)

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "mom": mom_data
    })

@app.route('/api/set_target_language', methods=['POST'])
def set_target_language():
    data = request.get_json() or {}
    lang_code = data.get("lang_code", "en")
    session_manager.set_target_language(lang_code)
    return jsonify({
        "status": "success",
        "target_lang": lang_code,
        "language_name": TARGET_LANGUAGES.get(lang_code, "English")
    })

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return jsonify({
        "status": "success",
        "filename": filename,
        "filepath": filepath
    })

@app.route('/api/stream_captions')
def stream_captions():
    q = session_manager.register_listener()

    def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_manager.session_id})}\n\n"
            while True:
                try:
                    data = q.get(timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except GeneratorExit:
            session_manager.unregister_listener(q)

    return Response(event_generator(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/api/list_sessions', methods=['GET'])
def list_sessions():
    recordings_dir = os.path.join(os.getcwd(), "recordings")
    sessions = []
    if os.path.exists(recordings_dir):
        for s in sorted(os.listdir(recordings_dir), reverse=True):
            mom_path = os.path.join(recordings_dir, s, "meeting_mom.json")
            if os.path.exists(mom_path):
                try:
                    with open(mom_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": s,
                        "utterances": data.get("total_utterances", 0),
                        "duration_sec": data.get("estimated_duration_sec", 0),
                        "summary": data.get("executive_summary", "")[:120] + "..."
                    })
                except Exception:
                    pass
    return jsonify({"sessions": sessions})

@app.route('/api/session_mom/<session_id>', methods=['GET'])
def get_session_mom(session_id):
    path = os.path.join(os.getcwd(), "recordings", session_id, "meeting_mom.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/download_mom/<session_id>/<fmt>', methods=['GET'])
def download_mom(session_id, fmt):
    session_dir = os.path.join(os.getcwd(), "recordings", session_id)
    if fmt == "html":
        filepath = os.path.join(session_dir, "meeting_mom.html")
    elif fmt == "md":
        filepath = os.path.join(session_dir, "meeting_mom.md")
    else:
        filepath = os.path.join(session_dir, "meeting_mom.json")

    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    print("=" * 70)
    print("  Real-time VR Captions, Diarization & MoM Generator Web Server")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
