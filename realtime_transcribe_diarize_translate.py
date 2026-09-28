"""
realtime_transcribe_diarize_translate.py
Near real-time microphone -> VAD -> faster-whisper -> pyannote diarization -> MarianMT translate -> live subtitles with speaker tags.
"""

import queue
import threading
import time
import os
import json

import numpy as np
import sounddevice as sd
import webrtcvad
import scipy.signal

# faster-whisper
from faster_whisper import WhisperModel

# transformers for translation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# pyannote for diarization
from pyannote.audio import Pipeline

# ----------------------------
# Config
# ----------------------------
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "medium"  # or "small", "large-v3", "tiny"
whisper_model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type="float32")



DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or False) else "cpu"  # pick "cuda" if available
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5  # seconds per audio read
VAD_MODE = 2  # 0-3 (aggressiveness)

# buffering for diarization (seconds)
DIARIZATION_WINDOW = 30.0

# translation: use MarianMT Helsinki models mapping for source->en when needed
# We'll dynamically load the appropriate Marian model based on detected language code.

# ----------------------------
# Audio capture -> queue
# ----------------------------
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Sounddevice status:", status)
    audio_q.put(indata.copy())

def start_microphone_stream():
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS
    stream = sd.InputStream(callback=audio_callback, blocksize=int(SAMPLE_RATE * BLOCK_DURATION))
    stream.start()
    return stream

# ----------------------------
# Utilities
# ----------------------------
def resample_if_needed(samples, orig_sr, target_sr=SAMPLE_RATE):
    if orig_sr == target_sr:
        return samples
    duration = samples.shape[0] / orig_sr
    new_len = int(duration * target_sr)
    resampled = scipy.signal.resample(samples, new_len)
    return resampled

# ----------------------------
# VAD-based chunking
# ----------------------------
vad = webrtcvad.Vad(VAD_MODE)

def frame_generator_from_queue():
    """Yield 10/20/30 ms frames from audio_q"""
    frame_duration_ms = 30
    frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
    buffer = np.zeros((0,), dtype='int16')
    while True:
        block = audio_q.get()
        if block is None:
            break
        # sounddevice gives float32 in [-1,1], convert to int16
        f = block.flatten()
        i16 = np.int16(np.clip(f * 32768, -32768, 32767))
        buffer = np.concatenate([buffer, i16])
        # yield frames of frame_size
        while len(buffer) >= frame_size:
            frame = buffer[:frame_size]
            buffer = buffer[frame_size:]
            yield frame.tobytes()

def vad_collector(generator, padding_ms=300, ratio=0.75):
    """Collect voiced chunks (simple VAD aggregator). Yields raw PCM bytes for each speech chunk."""
    frame_duration_ms = 30
    num_padding_frames = int(padding_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False
    voiced_frames = []
    for frame in generator:
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            # if enough voiced frames become true
            if sum(1 for f in ring_buffer if vad.is_speech(f, SAMPLE_RATE)) > ratio * len(ring_buffer):
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            if sum(1 for f in ring_buffer if not vad.is_speech(f, SAMPLE_RATE)) > ratio * len(ring_buffer):
                # end of utterance
                yield b"".join(voiced_frames)
                ring_buffer = []
                voiced_frames = []
                triggered = False

# ----------------------------
# Initialize models
# ----------------------------
print("Loading faster-whisper model from", MODEL_PATH)
whisper_model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type="float32")  # adjust compute_type as needed

print("Loading pyannote diarization pipeline (may require HF token)...")
# Set HF_TOKEN in the environment, or authenticate once with `huggingface-cli login`.
hf_token = os.getenv("HF_TOKEN")
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,
)



# translation cache (language->pipeline)
translation_pipelines = {}

def get_translation_pipeline(src_lang_code):
    # Marian models use codes like 'Helsinki-NLP/opus-mt-<src>-en'
    model_name = f"Helsinki-NLP/opus-mt-{src_lang_code}-en"
    if src_lang_code in translation_pipelines:
        return translation_pipelines[src_lang_code]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)
        translation_pipelines[src_lang_code] = translator
        return translator
    except Exception as e:
        print("No Marian model found for", src_lang_code, " — will skip translation. Error:", e)
        return None

# ----------------------------
# Main processing threads
# ----------------------------
transcript_buffer = []  # list of (start, end, speaker, text, lang)
diar_audio_buffer = bytearray()  # keep rolling audio for diarization (last DIARIZATION_WINDOW seconds)
diar_lock = threading.Lock()

def process_speech_chunks():
    frames = frame_generator_from_queue()
    for speech_pcm in vad_collector(frames):
        # convert PCM bytes -> numpy float32
        pcm = np.frombuffer(speech_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # append to rolling diarization buffer
        with diar_lock:
            diar_audio_buffer.extend(speech_pcm)
            # keep only recent DIARIZATION_WINDOW seconds
            max_bytes = int(DIARIZATION_WINDOW * SAMPLE_RATE * 2)  # 2 bytes per sample
            if len(diar_audio_buffer) > max_bytes:
                diar_audio_buffer[:] = diar_audio_buffer[-max_bytes:]

        # --- Transcribe this chunk with faster-whisper ---
        # faster-whisper accepts file path or numpy array; we'll use numpy array
        print("Transcribing chunk (len secs):", len(pcm)/SAMPLE_RATE)
        segments, info = whisper_model.transcribe(pcm, beam_size=5, language=None, vad_filter=False)  # returns generator for segments
        # note: adjust args as per faster-whisper API version
        # collect full text and language
        full_text = " ".join([seg.text for seg in segments])
        detected_lang = info.language if hasattr(info, "language") else None
        print("[ASR] lang=", detected_lang, "text=", full_text[:120])

        # If language not English, translate
        translated_text = full_text
        if detected_lang and detected_lang != "en":
            # map whisper codes to marian code (whisper uses full names; assumption: code same as ISO)
            translator = get_translation_pipeline(detected_lang)
            if translator:
                try:
                    res = translator(full_text, max_length=400)
                    translated_text = res[0]["translation_text"]
                except Exception as e:
                    print("Translation error:", e)

        # collect transcript with approximate timestamp (we don't have absolute time here; we can use time.time())
        now = time.time()
        transcript_item = {"time": now, "text": translated_text, "orig_lang": detected_lang or "unknown"}
        transcript_buffer.append(transcript_item)

        # print as live subtitle (speaker label will be added after diarization assignment)
        print(f"[LIVE] {time.strftime('%H:%M:%S', time.localtime(now))} : {translated_text}")

def run_periodic_diarization(interval=5.0):
    """Every `interval` seconds, run diarization on the current rolling buffer and assign speaker segments to transcripts."""
    while True:
        time.sleep(interval)
        with diar_lock:
            audio_bytes = bytes(diar_audio_buffer)
        if len(audio_bytes) < SAMPLE_RATE * 2 * 1:  # less than 1 second
            continue
        # Save rolling buffer to temporary wav
        import wave, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        try:
            # run diarization pipeline
            print("Running diarization on recent buffer (this may take a while)...")
            diarization = diar_pipeline(tmp.name)
            # diarization is an annotation of time segments with 'speaker' labels
            # assign speaker labels to recent transcript items by time proximity (approx match)
            # NOTE: diar_pipeline timestamps are relative to wav start; we used rolling buffer without absolute time;
            # For a robust production solution you'd maintain absolute timestamps; this example uses approximate matching.
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                # find transcript entries with times close to now - buffer_length + start
                # approximate mapping: buffer_end_time = now; buffer_start_time = now - buffer_length
                buffer_length = len(audio_bytes) / (SAMPLE_RATE * 2)
                buffer_end_time = time.time()
                buffer_start_time = buffer_end_time - buffer_length
                seg_start_abs = buffer_start_time + start
                seg_end_abs = buffer_start_time + end
                # attach speaker to transcripts in range
                for t in transcript_buffer:
                    if seg_start_abs - 0.5 <= t["time"] <= seg_end_abs + 0.5:
                        t["speaker"] = speaker
            # print updated subtitles with speaker labels
            print("----- Subtitles (with speaker tags) -----")
            for t in transcript_buffer[-20:]:
                sp = t.get("speaker", "SPEAKER_?") 
                ts = time.strftime("%H:%M:%S", time.localtime(t["time"]))
                print(f"[{ts}] {sp}: {t['text']}")
            print("-----------------------------------------")
        except Exception as e:
            print("Diarization error:", e)
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

# ----------------------------
# Start everything
# ----------------------------
if __name__ == "__main__":
    print("Starting microphone...")
    stream = start_microphone_stream()
    t1 = threading.Thread(target=process_speech_chunks, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_periodic_diarization, daemon=True)
    t2.start()
    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        audio_q.put(None)
        stream.stop()
        stream.close()
