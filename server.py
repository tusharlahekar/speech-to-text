
import asyncio
import websockets
import json
import base64
import pickle
import os
import torch
import numpy as np
import librosa
import nest_asyncio
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive
import ssl
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
nest_asyncio.apply()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_RATE           = 8000
SPEECH_THRESHOLD      = 0.5
SILENCE_LIMIT         = 20
MIN_SPEECH_CHUNKS     = 5
SIMILARITY_THRESHOLD  = 0.75
INTENT_CONF_THRESHOLD = 0.60

SPEAKER_LABEL = {"SPEAKER_00": "AGENT", "SPEAKER_01": "CALLER"}

dashboard_clients = set()
active_calls      = {}
all_transcripts   = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == "cpu":
    print("   ⚠️  No GPU — Runtime → Change runtime type → T4 GPU\n")

# ─── ENTITY EXTRACTION ────────────────────────────────────────────────────────
_VEH_RE     = re.compile(r'\b([A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4})\b', re.I)
_ORDER_RE   = re.compile(r'\b(\d{5,12})\b')
_INVOICE_RE = re.compile(r'\b(?:INV|INVOICE|BILL|RECEIPT)[- ]?([A-Z0-9\-]{4,15})\b', re.I)

PART_KEYWORDS = [
    "brake pad", "brake disc", "brake shoe", "clutch plate", "clutch kit",
    "air filter", "oil filter", "fuel filter", "cabin filter",
    "engine oil", "gear oil", "transmission oil",
    "shock absorber", "strut", "suspension",
    "battery", "alternator", "starter motor",
    "radiator", "cooling fan", "thermostat", "coolant",
    "fuel pump", "injector", "carburettor",
    "ignition coil", "spark plug", "distributor cap",
    "timing belt", "timing chain", "timing kit",
    "water pump", "fan belt", "serpentine belt", "drive belt",
    "wheel bearing", "hub bearing", "cv axle", "cv joint", "drive shaft",
    "tie rod", "ball joint", "control arm",
    "brake caliper", "master cylinder", "brake booster",
    "wiper blade", "wiper motor",
    "headlight", "tail light", "fog light", "indicator light",
    "bumper", "fender", "bonnet", "hood", "door handle", "side mirror",
    "exhaust", "muffler", "catalytic converter",
    "piston", "piston ring", "cylinder liner",
    "gasket", "head gasket", "intake manifold gasket",
    "camshaft", "crankshaft", "connecting rod",
    "gearbox", "transmission", "differential",
    "steering rack", "power steering pump", "rack and pinion",
    "ac compressor", "ac condenser", "blower motor",
    "wheel rim", "tyre", "tube",
]

def extract_entities(text: str) -> dict:
    vehicle_no = None
    m = _VEH_RE.search(text)
    if m:
        vehicle_no = re.sub(r'[\s\-]', '', m.group(0)).upper()

    order_no = None
    for m in _ORDER_RE.finditer(text):
        if len(m.group(1)) >= 5:
            order_no = m.group(1)
            break

    invoice_no = None
    m = _INVOICE_RE.search(text)
    if m:
        invoice_no = m.group(0)

    parts = [kw for kw in PART_KEYWORDS if kw in text.lower()]

    return {
        "vehicle_no": vehicle_no,
        "order_no":   order_no,
        "invoice_no": invoice_no,
        "parts":      parts,
    }


# ─── PER-CALL CONTEXT ─────────────────────────────────────────────────────────
@dataclass
class CallContext:
    vehicle_no:   Optional[str] = None
    order_no:     Optional[str] = None
    invoice_no:   Optional[str] = None
    parts:        list = field(default_factory=list)
    recent_turns: list = field(default_factory=list)

    def update(self, entities: dict, speaker: str, text: str):
        if entities["vehicle_no"]:
            self.vehicle_no = entities["vehicle_no"]
        if entities["order_no"]:
            self.order_no = entities["order_no"]
        if entities["invoice_no"]:
            self.invoice_no = entities["invoice_no"]
        for p in entities["parts"]:
            if p not in self.parts:
                self.parts.append(p)
        self.recent_turns.append((speaker, text))
        if len(self.recent_turns) > 4:
            self.recent_turns.pop(0)

    def summary(self) -> str:
        lines = []
        if self.vehicle_no:
            lines.append(f"Vehicle Number: {self.vehicle_no}")
        if self.order_no:
            lines.append(f"Order Number: #{self.order_no}")
        if self.invoice_no:
            lines.append(f"Invoice: {self.invoice_no}")
        if self.parts:
            lines.append(f"Parts mentioned: {', '.join(self.parts)}")
        if self.recent_turns:
            lines.append("Recent conversation:")
            for spk, txt in self.recent_turns[:-1]:
                lines.append(f"  {spk}: {txt}")
        return "\n".join(lines) if lines else "No entities captured yet."


# ─── LOAD INTENT MODEL FROM GOOGLE DRIVE ─────────────────────────────────────
print("⏳ Mounting Google Drive...")
drive.mount('/content/drive')
PICKLE_PATH = "/content/drive/MyDrive/trained_intent_model.pkl"

if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(
        f"\n❌ Intent model not found at: {PICKLE_PATH}\n"
        "   Run Cell 1 first to train and save the model."
    )

print("⏳ Loading intent model from Google Drive...")
with open(PICKLE_PATH, "rb") as f:
    _bundle = pickle.load(f)

intent_encoder = _bundle["encoder"]
_clf           = _bundle["classifier"]
_le            = _bundle["label_encoder"]
cv_acc         = _bundle.get("cv_accuracy", 0)
enc_name       = _bundle.get("encoder_name", "MiniLM")
print(
    f"✅ Intent model loaded\n"
    f"   Intents:  {len(_bundle['intent_names'])}\n"
    f"   Examples: {_bundle['num_examples']}\n"
    f"   Encoder:  {enc_name}\n"
    f"   CV Acc:   {cv_acc:.1%}\n"
    f"   Trained:  {_bundle['trained_at']}\n"
)

# ─── SILERO VAD ───────────────────────────────────────────────────────────────
print("⏳ Loading Silero VAD...")
vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
vad_model.eval()
print("✅ VAD loaded")

print("⏳ Loading Resemblyzer speaker encoder (~17MB)...")
voice_encoder = VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
print("✅ Resemblyzer loaded")

# ─── WHISPER LARGE-V3 ─────────────────────────────────────────────────────────
print("⏳ Loading Whisper large-v3...")
whisper_model = WhisperModel(
    "large-v3",
    device=str(device),
    compute_type="int8_float16" if device.type == "cuda" else "int8",
)
print("✅ Whisper loaded\n")

# ─── QWEN 2.5-1.5B ───────────────────────────────────────────────────────────
print("⏳ Loading Qwen2.5-1.5B-Instruct (float16, ~3GB)...")
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
qwen_tok   = AutoTokenizer.from_pretrained(QWEN_MODEL)
qwen_mdl   = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL, dtype=torch.float16, device_map="auto",
)
qwen_mdl.eval()
print("✅ Qwen2.5-1.5B-Instruct loaded\n")

if device.type == "cuda":
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"📊 VRAM: {used:.1f}GB / {total:.1f}GB  ({total-used:.1f}GB free)\n")

LANG_MAP = {
    "en": "English", "hi": "Hindi", "mr": "Marathi",
    "gu": "Gujarati", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "bn": "Bengali", "pa": "Punjabi",
}

QWEN_SYSTEM = """You are a helpful customer service agent for an automotive spare parts ecommerce platform in India.

RULES:
1. Reply ONLY in the SAME language the customer used. English → English. Hindi → Hindi. Never switch.
2. Keep reply SHORT: 2 sentences maximum. This is a live phone call.
3. If a vehicle number, order number, or invoice number was captured, reference it directly in your reply.
4. If you do not have real system data, say "Let me check that in our system right now."
5. Do NOT repeat the customer's question. Be action-oriented."""


# ─── INTENT DETECTION ─────────────────────────────────────────────────────────
def detect_intent(text: str) -> dict:
    t0   = time.time()
    emb  = intent_encoder.encode([text], show_progress_bar=False)
    pred = _clf.predict(emb)[0]
    prob = _clf.predict_proba(emb)[0]
    name = _le.inverse_transform([pred])[0]
    conf = round(float(max(prob)), 3)
    return {
        "intent":     name,
        "confidence": conf,
        "intent_ms":  int((time.time() - t0) * 1000),
    }


# ─── QWEN INFERENCE ───────────────────────────────────────────────────────────
def run_qwen(text: str, intent: str, lang: str, ctx: CallContext) -> tuple[str, int]:
    t0        = time.time()
    lang_name = LANG_MAP.get(lang, "Hindi")
    ctx_text  = ctx.summary()

    user_msg = (
        f"Customer language: {lang_name}\n"
        f"Customer said: \"{text}\"\n"
        f"Detected intent: {intent}\n"
        f"\nCall context:\n{ctx_text}\n"
        f"\nReply in {lang_name}:"
    )

    messages = [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    fmt    = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tok([fmt], return_tensors="pt").to(qwen_mdl.device)

    with torch.no_grad():
        out = qwen_mdl.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=qwen_tok.eos_token_id,
        )

    new_ids  = out[0][inputs.input_ids.shape[1]:]
    response = qwen_tok.decode(new_ids, skip_special_tokens=True).strip()

    for stop in ["Customer language:", "Customer said:", "Detected intent:",
                 "Call context", "Reply in", "RULES:"]:
        if stop in response:
            response = response[:response.index(stop)].strip()

    return response, int((time.time() - t0) * 1000)


# ─── BROADCAST ────────────────────────────────────────────────────────────────
async def broadcast(payload: dict):
    global dashboard_clients
    if not dashboard_clients:
        return
    msg  = json.dumps(payload, ensure_ascii=False)
    dead = set()
    for ws in list(dashboard_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    dashboard_clients -= dead


def public_calls():
    return [{k: v for k, v in c.items() if not k.startswith("_")}
            for c in active_calls.values()]


class RealtimeSpeakerTracker:
    def __init__(self, threshold=SIMILARITY_THRESHOLD, max_speakers=2):
        self.threshold    = threshold
        self.max_speakers = max_speakers
        self.centroids: dict[str, np.ndarray] = {}
        self._count       = 0

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def identify(self, audio_16k: np.ndarray) -> str:
        """
        audio_16k: float32 numpy array at 16kHz (already resampled by pipeline)
        Returns: "SPEAKER_00" or "SPEAKER_01"
        """

        wav = preprocess_wav(audio_16k, source_sr=16000)

        embed = voice_encoder.embed_utterance(wav)

        if not self.centroids:
            self.centroids["SPEAKER_00"] = embed
            self._count = 1
            return "SPEAKER_00"

        best_id, best_sim = None, -1.0
        for spk_id, centroid in self.centroids.items():
            sim = self._cosine_sim(embed, centroid)
            if sim > best_sim:
                best_sim, best_id = sim, spk_id

        if best_sim >= self.threshold:
            self.centroids[best_id] = 0.9 * self.centroids[best_id] + 0.1 * embed
            return best_id

        if self._count < self.max_speakers:
            new_id = f"SPEAKER_0{self._count}"
            self.centroids[new_id] = embed
            self._count += 1
            return new_id

        return best_id


# ─── VAD BUFFER ───────────────────────────────────────────────────────────────
class TrackProcessor:
    def __init__(self, name: str):
        self.name            = name
        self.internal_buffer = bytearray()
        self.speech_buffer   = bytearray()
        self.silence_count   = 0
        self.is_speaking     = False
        self.speech_chunks   = 0

    def _is_speech(self, frame: bytes) -> bool:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        return vad_model(torch.from_numpy(audio), SAMPLE_RATE).item() >= SPEECH_THRESHOLD

    def process_chunk(self, new_bytes: bytes) -> list[bytes]:
        self.internal_buffer.extend(new_bytes)
        completed = []
        FRAME = 512
        while len(self.internal_buffer) >= FRAME:
            frame = self.internal_buffer[:FRAME]
            del self.internal_buffer[:FRAME]
            if self._is_speech(frame):
                if not self.is_speaking:
                    print("🟢 VAD: speech started")
                self.is_speaking    = True
                self.silence_count  = 0
                self.speech_chunks += 1
                self.speech_buffer.extend(frame)
            else:
                if self.is_speaking:
                    self.silence_count += 1
                    self.speech_buffer.extend(frame)
                    if self.silence_count >= SILENCE_LIMIT:
                        if self.speech_chunks >= MIN_SPEECH_CHUNKS:
                            completed.append(bytes(self.speech_buffer))
                        self._reset()
        return completed

    def flush(self) -> bytes | None:
        if self.speech_buffer and self.speech_chunks >= MIN_SPEECH_CHUNKS:
            data = bytes(self.speech_buffer)
            self._reset()
            return data
        self._reset()
        return None

    def _reset(self):
        self.speech_buffer = bytearray()
        self.silence_count = 0
        self.is_speaking   = False
        self.speech_chunks = 0


# ─── CORE PIPELINE ────────────────────────────────────────────────────────────
async def identify_and_transcribe(
    pcm_8k:     bytes,
    tracker:    RealtimeSpeakerTracker,
    stream_sid: str,
    call_ctx:   CallContext,
):
    global active_calls, all_transcripts
    print(f"🔴 VAD: pause — {len(pcm_8k):,} bytes")

    def _run():
        t0        = time.time()
        audio_8k  = np.frombuffer(pcm_8k, dtype=np.int16).astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_8k, orig_sr=8000, target_sr=16000)

        raw_id   = tracker.identify(audio_16k)
        label    = SPEAKER_LABEL.get(raw_id, raw_id)
        sim_info = {k: f"{tracker._cosine_sim(tracker.centroids[raw_id], v):.2f}"
                    for k, v in tracker.centroids.items()}
        print(f"   🆔 Speaker: {label}  (sim: {sim_info})")

        segs, info = whisper_model.transcribe(
            audio_16k,
            language=None,
            beam_size=5,
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        valid  = [s.text for s in segs if getattr(s, "no_speech_prob", 0) < 0.6]
        text   = " ".join(valid).strip()
        stt_ms = int((time.time() - t0) * 1000)
        return label, text, info.language, stt_ms

    label, text, lang, stt_ms = await asyncio.to_thread(_run)

    if not text:
        print(f"   ⚠️  [{label}] — no words (noise/breathing)\n")
        return

    print(f"🗣️  [{label}] ({lang}) [{stt_ms}ms]: {text}\n")

    entities = await asyncio.to_thread(extract_entities, text)
    call_ctx.update(entities, label, text)

    found = {k: v for k, v in entities.items() if v}
    if found:
        print(f"   📌 Entities: {found}")
    if call_ctx.vehicle_no:
        print(f"   🚗 Call vehicle: {call_ctx.vehicle_no}")
    if call_ctx.order_no:
        print(f"   📦 Call order:   #{call_ctx.order_no}")

    intent_data = await asyncio.to_thread(detect_intent, text)
    print(
        f"   🎯 [{label}] {intent_data['intent']} "
        f"({intent_data['confidence']:.0%}) | {intent_data['intent_ms']}ms"
    )

    suggestion = ""
    qwen_ms    = 0

    if (label == "CALLER"
            and intent_data["intent"] != "other"
            and intent_data["confidence"] >= INTENT_CONF_THRESHOLD):
        print(f"   🤖 Qwen generating ({LANG_MAP.get(lang, lang)})...")
        suggestion, qwen_ms = await asyncio.to_thread(
            run_qwen, text, intent_data["intent"], lang, call_ctx
        )
        print(f"   💬 [{qwen_ms}ms]: {suggestion}\n")
    elif label == "CALLER" and intent_data["intent"] != "other":
        print(f"   ⏭️  Conf {intent_data['confidence']:.0%} < threshold — Qwen skipped\n")
    else:
        print()

    now     = datetime.now().strftime("%H:%M:%S")
    line_id = f"{stream_sid}_{int(time.time()*1000)}"
    entry   = {
        "type":        "transcript",
        "line_id":     line_id,
        "call_sid":    stream_sid,
        "time":        now,
        "speaker":     label,
        "lang":        lang,
        "text":        text,
        "stt_ms":      stt_ms,
        "intent":      intent_data["intent"],
        "confidence":  intent_data["confidence"],
        "intent_ms":   intent_data["intent_ms"],
        "vehicle_no":  entities.get("vehicle_no"),
        "order_id":    entities.get("order_no") or call_ctx.order_no,
        "invoice_no":  entities.get("invoice_no"),
        "parts":       entities.get("parts", []),
        "ctx_vehicle": call_ctx.vehicle_no,
        "ctx_order":   call_ctx.order_no,
        "suggestion":  suggestion,
        "qwen_ms":     qwen_ms,
    }

    all_transcripts.append(entry)
    if stream_sid in active_calls:
        active_calls[stream_sid]["turns"]   += 1
        active_calls[stream_sid]["duration"] = int(
            time.time() - active_calls[stream_sid]["_start"]
        )

    await broadcast(entry)
    await broadcast({"type": "calls", "calls": public_calls()})


# ─── EXOTEL HANDLER ───────────────────────────────────────────────────────────
async def exotel_ws(websocket):
    global active_calls

    stream_sid = None
    start_ts   = time.time()
    call_ctx   = CallContext()

    print("📞 New call — waiting for start event...")
    vad     = TrackProcessor("stream")
    tracker = RealtimeSpeakerTracker()

    try:
        async for message in websocket:
            data  = json.loads(message)
            event = data.get("event", "")

            if event == "start":
                stream_sid = (
                    data.get("stream_sid") or data.get("streamSid")
                    or data.get("callSid") or f"call_{int(time.time())}"
                )
                print(f"📞 Stream started: {stream_sid}")
                active_calls[stream_sid] = {
                    "call_sid":   stream_sid,
                    "start_time": datetime.now().strftime("%H:%M:%S"),
                    "status":     "live",
                    "turns":      0,
                    "duration":   0,
                    "_start":     start_ts,
                }
                await broadcast({
                    "type": "call_start",
                    "call": {k: v for k, v in active_calls[stream_sid].items()
                             if not k.startswith("_")},
                })
                await broadcast({"type": "calls", "calls": public_calls()})

            elif event == "media":
                if stream_sid is None:
                    stream_sid = (
                        data.get("stream_sid") or data.get("streamSid")
                        or f"call_{int(time.time())}"
                    )
                    print(f"📞 Stream (from media): {stream_sid}")
                    active_calls[stream_sid] = {
                        "call_sid":   stream_sid,
                        "start_time": datetime.now().strftime("%H:%M:%S"),
                        "status":     "live", "turns": 0, "duration": 0,
                        "_start":     start_ts,
                    }
                    await broadcast({
                        "type": "call_start",
                        "call": {k: v for k, v in active_calls[stream_sid].items()
                                 if not k.startswith("_")},
                    })
                    await broadcast({"type": "calls", "calls": public_calls()})

                pcm = base64.b64decode(data["media"]["payload"])
                for utterance in vad.process_chunk(pcm):
                    await identify_and_transcribe(utterance, tracker, stream_sid, call_ctx)

            elif event == "stop":
                print(f"\n🛑 Call ended — flushing buffer")
                remaining = vad.flush()
                if remaining and stream_sid:
                    await identify_and_transcribe(remaining, tracker, stream_sid, call_ctx)
                print(f"\n📋 Call entity summary:")
                print(f"   Vehicle: {call_ctx.vehicle_no or 'not captured'}")
                print(f"   Order:   {call_ctx.order_no   or 'not captured'}")
                print(f"   Invoice: {call_ctx.invoice_no or 'not captured'}")
                if call_ctx.parts:
                    print(f"   Parts:   {', '.join(call_ctx.parts)}")
                print("─" * 50)
                break

    except Exception as e:
        print(f"⚠️  Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        dur = int(time.time() - start_ts)
        if stream_sid and stream_sid in active_calls:
            active_calls[stream_sid]["status"]   = "ended"
            active_calls[stream_sid]["duration"] = dur
        if stream_sid:
            await broadcast({"type": "call_end", "call_sid": stream_sid, "duration": dur})
            await broadcast({"type": "calls", "calls": public_calls()})
        print(f"📊 {stream_sid} ended | {dur}s")


# ─── DASHBOARD HANDLER ────────────────────────────────────────────────────────
async def dashboard_handler(websocket):
    global dashboard_clients
    dashboard_clients.add(websocket)
    n = len(dashboard_clients)
    print(f"🖥️  Dashboard connected ({n} tab{'s' if n > 1 else ''})")
    try:
        await websocket.send(json.dumps({
            "type":        "init",
            "calls":       public_calls(),
            "transcripts": all_transcripts[-500:],
        }, ensure_ascii=False))
        await websocket.wait_closed()
    except Exception:
        pass
    finally:
        dashboard_clients.discard(websocket)
        print(f"🖥️  Dashboard disconnected ({len(dashboard_clients)} remaining)")


async def router(websocket):
    try:
        path = websocket.request.path
    except AttributeError:
        path = getattr(websocket, "path", "/")
    if "/dashboard" in path:
        await dashboard_handler(websocket)
    else:
        await exotel_ws(websocket)


async def main():
    server = await websockets.serve(
        router, "0.0.0.0", 8000,
        ping_interval=None, max_size=10 * 1024 * 1024,
    )
    print("=" * 60)
    print("🚀  server running on port 8000")
    print("    Exotel StreamUrl  →  wss://YOUR_NGROK/")
    print("    Dashboard WS      →  wss://YOUR_NGROK/dashboard")
    print("=" * 60)
    await server.wait_closed()

asyncio.run(main())
