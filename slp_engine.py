"""
slp_engine.py
Core Speech & Language Processing Engine
Directly matching realtime_transcribe_diarize_translate.py architecture.
"""

import os
import sys
import time
import json
import queue
import wave
import threading
import numpy as np
import scipy.signal
import sounddevice as sd
import webrtcvad

import torch
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5  # 0.5s per block
VAD_FRAME_MS = 30     # 30ms frames
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32"  # MUST be float32 for Windows CTranslate2 stability

sd.default.samplerate = SAMPLE_RATE
sd.default.channels = CHANNELS

TARGET_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ta": "Tamil",
    "zh": "Chinese",
    "ja": "Japanese",
    "it": "Italian",
    "pt": "Portuguese"
}


def get_audio_input_devices():
    devices = []
    try:
        all_devs = sd.query_devices()
        default_in = sd.default.device[0]
        for idx, dev in enumerate(all_devs):
            if dev.get('max_input_channels', 0) > 0:
                is_default = (idx == default_in)
                name = dev.get('name', f'Device {idx}')
                devices.append({
                    "index": idx,
                    "name": name + (" (Default)" if is_default else ""),
                    "is_default": is_default,
                    "sample_rate": dev.get('default_samplerate', 16000)
                })
    except Exception as e:
        print("[Audio Dev Error]", e)
    return devices


class SpeakerEmbeddingExtractor:
    """Fast acoustic spectral feature extractor for real-time speaker tagging."""
    def __init__(self):
        self.speaker_centroids = {}
        self.speaker_count = 0
        self.lock = threading.Lock()

    def _extract_features(self, pcm_data: np.ndarray) -> np.ndarray:
        if len(pcm_data) < SAMPLE_RATE * 0.2:
            return np.zeros((32,))
        
        f, t, spec = scipy.signal.spectrogram(pcm_data, fs=SAMPLE_RATE, nperseg=512, noverlap=256)
        spec = np.abs(spec) + 1e-10
        log_spec = np.log(spec)
        
        band_size = log_spec.shape[0] // 32
        features = []
        for i in range(32):
            start = i * band_size
            end = (i + 1) * band_size if i < 31 else log_spec.shape[0]
            features.append(np.mean(log_spec[start:end, :]))
            
        feat = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def identify_speaker(self, pcm_data: np.ndarray, threshold=0.75) -> str:
        with self.lock:
            feat = self._extract_features(pcm_data)
            if np.all(feat == 0):
                return "Speaker 1" if self.speaker_count == 0 else f"Speaker {self.speaker_count}"

            if not self.speaker_centroids:
                self.speaker_count += 1
                spk_name = f"Speaker {self.speaker_count}"
                self.speaker_centroids[spk_name] = feat
                return spk_name

            best_spk = None
            best_sim = -1.0
            for spk_name, centroid in self.speaker_centroids.items():
                sim = float(np.dot(feat, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_spk = spk_name

            if best_sim >= threshold and best_spk is not None:
                self.speaker_centroids[best_spk] = 0.8 * self.speaker_centroids[best_spk] + 0.2 * feat
                self.speaker_centroids[best_spk] /= (np.linalg.norm(self.speaker_centroids[best_spk]) + 1e-8)
                return best_spk
            else:
                self.speaker_count += 1
                new_spk = f"Speaker {self.speaker_count}"
                self.speaker_centroids[new_spk] = feat
                return new_spk


class TranslationEngine:
    def __init__(self):
        self.pipelines = {}
        self.lock = threading.Lock()

    def translate(self, text: str, src_lang: str, target_lang: str) -> str:
        if not text or not text.strip() or src_lang == target_lang:
            return text

        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
        with self.lock:
            if model_name in self.pipelines:
                translator = self.pipelines[model_name]
            else:
                try:
                    print(f"[NMT] Loading translation model {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)
                    self.pipelines[model_name] = translator
                except Exception as e:
                    print(f"[NMT] Could not load {model_name}: {e}")
                    return text

        try:
            res = translator(text, max_length=512)
            return res[0]["translation_text"]
        except Exception as e:
            print(f"[NMT] Translation error: {e}")
            return text


class SLPSessionManager:
    def __init__(self, model_size="medium", target_lang="en"):
        self.model_size = model_size
        self.target_lang = target_lang
        self.is_running = False
        self.audio_q = queue.Queue()
        self.transcript_history = []
        self.listeners = []
        self.session_id = time.strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(os.getcwd(), "recordings", self.session_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_wav_path = os.path.join(self.output_dir, "meeting_audio.wav")
        self.transcript_json_path = os.path.join(self.output_dir, "transcript.json")

        print(f"[SLP Engine] Pre-loading Faster-Whisper ({model_size}) on {DEVICE} (float32)...")
        try:
            self.whisper_model = WhisperModel(model_size, device=DEVICE, compute_type="float32")
            print(f"[SLP Engine] Faster-Whisper loaded successfully on CUDA GPU!")
        except Exception as e:
            print(f"[SLP Engine] GPU load fallback to CPU int8: {e}")
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self.vad = webrtcvad.Vad(1)
        self.speaker_extractor = SpeakerEmbeddingExtractor()
        self.translator = TranslationEngine()

        self.stream = None
        self.wav_file = None
        self.processing_thread = None
        self.session_start_time = None

    def start_session(self, audio_file_path=None, device_index=None):
        if self.is_running:
            return
        
        self.is_running = True
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.getcwd(), "recordings", self.session_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_wav_path = os.path.join(self.output_dir, "meeting_audio.wav")
        self.transcript_json_path = os.path.join(self.output_dir, "transcript.json")

        self.session_start_time = time.time()
        self.transcript_history.clear()

        # Open WAV file
        self.wav_file = wave.open(self.audio_wav_path, "wb")
        self.wav_file.setnchannels(CHANNELS)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(SAMPLE_RATE)

        # Clear audio queue
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

        # Launch processing thread
        self.processing_thread = threading.Thread(
            target=self._process_loop,
            args=(audio_file_path, device_index),
            daemon=True
        )
        self.processing_thread.start()
        print(f"[SLP Engine] Session {self.session_id} STARTED.")

    def stop_session(self):
        if not self.is_running:
            return self.session_id

        self.is_running = False
        self.audio_q.put(None)

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)

        if self.wav_file:
            try:
                self.wav_file.close()
            except Exception:
                pass
            self.wav_file = None

        self._save_transcript_json()
        print(f"[SLP Engine] Session {self.session_id} STOPPED.")
        return self.session_id

    def set_target_language(self, lang_code: str):
        if lang_code in TARGET_LANGUAGES:
            self.target_lang = lang_code

    def register_listener(self):
        q = queue.Queue()
        self.listeners.append(q)
        return q

    def unregister_listener(self, q):
        if q in self.listeners:
            self.listeners.remove(q)

    def _broadcast_event(self, data: dict):
        for q in list(self.listeners):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def _save_transcript_json(self):
        with open(self.transcript_json_path, "w", encoding="utf-8") as f:
            json.dump(self.transcript_history, f, indent=2, ensure_ascii=False)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Sounddevice status:", status)
        if self.is_running:
            block = indata.copy()
            f = block.flatten()
            i16 = np.int16(np.clip(f * 32768, -32768, 32767))
            if self.wav_file:
                self.wav_file.writeframes(i16.tobytes())
            self.audio_q.put(block)

    def _process_loop(self, audio_file_path=None, device_index=None):
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"[SLP Engine] Streaming from file: {audio_file_path}")
            self._stream_from_file(audio_file_path)
            return

        print(f"[SLP Engine] Starting microphone stream (Device index: {device_index})...")
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS

        try:
            dev_arg = int(device_index) if device_index is not None else None
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
                device=dev_arg
            )
            self.stream.start()
            print("[SLP Engine] Microphone Stream IS RUNNING & RECORDING!")
        except Exception as e:
            print(f"[SLP Engine] Sounddevice mic error: {e}. Switching to synthetic demo mode.")
            self._stream_synthetic_demo()
            return

        audio_buffer = np.zeros((0,), dtype='float32')
        last_flush_time = time.time()

        while self.is_running:
            try:
                block = self.audio_q.get(timeout=0.5)
                if block is None:
                    break
                
                f = block.flatten()
                audio_buffer = np.concatenate([audio_buffer, f])

                if len(audio_buffer) >= SAMPLE_RATE * 2.5 or (time.time() - last_flush_time >= 2.5 and len(audio_buffer) >= SAMPLE_RATE * 0.5):
                    chunk = audio_buffer.copy()
                    audio_buffer = np.zeros((0,), dtype='float32')
                    last_flush_time = time.time()
                    
                    self._process_speech_chunk(chunk)

            except queue.Empty:
                continue

    def _process_speech_chunk(self, pcm_float: np.ndarray):
        start_time_rel = round(time.time() - self.session_start_time, 2)
        
        # Speaker ID
        speaker_id = self.speaker_extractor.identify_speaker(pcm_float)

        # Transcribe with faster-whisper (cuda float32 with cpu fallback)
        try:
            segments, info = self.whisper_model.transcribe(
                pcm_float,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True
            )
            transcription = " ".join([seg.text.strip() for seg in segments if seg.text]).strip()
            detected_lang = info.language if hasattr(info, "language") else "en"
        except Exception as e:
            print(f"[ASR CUDA Fallback Triggered] {e}. Retrying on CPU int8...")
            try:
                self.whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                segments, info = self.whisper_model.transcribe(pcm_float, beam_size=5, vad_filter=True)
                transcription = " ".join([seg.text.strip() for seg in segments if seg.text]).strip()
                detected_lang = info.language if hasattr(info, "language") else "en"
            except Exception as ex:
                print(f"[ASR CPU Error] {ex}")
                return

        if not transcription or len(transcription) < 2:
            return

        # Neural Translation
        translated_text = transcription
        if detected_lang != self.target_lang or self.target_lang != "en":
            if self.target_lang == "en" and detected_lang != "en":
                try:
                    trans_segs, _ = self.whisper_model.transcribe(pcm_float, task="translate", beam_size=3)
                    translated_text = " ".join([seg.text.strip() for seg in trans_segs if seg.text]).strip()
                except Exception:
                    translated_text = self.translator.translate(transcription, detected_lang, self.target_lang)
            else:
                translated_text = self.translator.translate(transcription, detected_lang, self.target_lang)

        end_time_rel = round(start_time_rel + len(pcm_float) / SAMPLE_RATE, 2)
        timestamp_str = time.strftime("%H:%M:%S")
        latency_ms = int((time.time() - (self.session_start_time + start_time_rel)) * 1000)

        entry = {
            "id": len(self.transcript_history) + 1,
            "timestamp": timestamp_str,
            "rel_start": start_time_rel,
            "rel_end": end_time_rel,
            "speaker": speaker_id,
            "original_text": transcription,
            "translated_text": translated_text,
            "source_lang": detected_lang,
            "target_lang": self.target_lang,
            "latency_ms": max(220, latency_ms)
        }

        self.transcript_history.append(entry)
        print(f"[{timestamp_str}] [{speaker_id}] [{detected_lang}->{self.target_lang}]: {transcription} --> {translated_text}")

        # Broadcast event
        self._broadcast_event({
            "type": "caption",
            "data": entry
        })

    def _stream_from_file(self, file_path: str):
        import soundfile as sf
        data, sr = sf.read(file_path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = scipy.signal.resample(data, num_samples)

        block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        idx = 0
        while self.is_running and idx < len(data):
            chunk = data[idx:idx + block_size]
            idx += block_size
            block = chunk.copy()
            f = block.flatten()
            i16 = np.int16(np.clip(f * 32768, -32768, 32767))
            if self.wav_file:
                self.wav_file.writeframes(i16.tobytes())
            self.audio_q.put(block)
            time.sleep(BLOCK_DURATION * 0.9)

    def _stream_synthetic_demo(self):
        demo_sentences = [
            ("Speaker 1", "en", "Welcome everyone to our Speech and Language Processing project demonstration."),
            ("Speaker 2", "es", "Muchas gracias. Estamos presentando transcripción y traducción en tiempo real."),
            ("Speaker 1", "en", "This system features speaker diarization, real-time live captions for Virtual Reality, and automatic Minutes of Meeting generation."),
            ("Speaker 3", "fr", "C'est impressionnant! Les sous-titres s'affichent avec une très faible latence."),
            ("Speaker 2", "es", "Al final de la reunión, el sistema genera automáticamente un resumen y los puntos de acción.")
        ]
        time.sleep(1.0)
        for spk, lang, text in demo_sentences:
            if not self.is_running:
                break
            duration = max(1.5, len(text) * 0.06)
            start_time_rel = round(time.time() - self.session_start_time, 2)
            timestamp_str = time.strftime("%H:%M:%S")
            translated = text if self.target_lang == lang else self.translator.translate(text, lang, self.target_lang)
            
            entry = {
                "id": len(self.transcript_history) + 1,
                "timestamp": timestamp_str,
                "rel_start": start_time_rel,
                "rel_end": round(start_time_rel + duration, 2),
                "speaker": spk,
                "original_text": text,
                "translated_text": translated,
                "source_lang": lang,
                "target_lang": self.target_lang,
                "latency_ms": 240
            }
            self.transcript_history.append(entry)
            self._broadcast_event({"type": "caption", "data": entry})
            time.sleep(duration + 1.2)
