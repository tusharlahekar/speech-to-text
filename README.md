# 🎙️ Real-Time Call Intelligence

A production-ready real-time speech-to-text and agent-assist system for phone calls.  
Audio streams from a telephony provider → VAD → speaker diarization → transcription → intent detection → LLM suggestion, all in under 3 seconds per utterance.

Built for multilingual Indian call centers supporting **English, Hindi, Marathi, and Hinglish (code-switched)**.

---

## ✨ Features

- **Real-time transcription** using Whisper large-v3 via `faster-whisper`
- **2-speaker diarization** using Resemblyzer d-vectors (no training required)
- **Voice Activity Detection** using Silero VAD — only processes actual speech
- **Multilingual intent classification** — 8 intents, trained on 700+ examples across 4 languages
- **LLM agent suggestions** — Qwen 2.5-1.5B generates context-aware replies in the caller's language
- **Entity extraction** — vehicle numbers, order IDs, invoice numbers, part names captured automatically
- **Live dashboard** — WebSocket-based JSON stream, easy to connect to any frontend

---

## 🏗️ Architecture

```
Telephony Provider (WebSocket audio stream)
        │
        ▼
┌──────────────────┐
│  Silero VAD      │  detects speech segments, discards silence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Resemblyzer     │  identifies AGENT vs CALLER per utterance
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Whisper v3      │  transcribes audio → text + language detection
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MiniLM + LR     │  classifies intent with confidence score
└────────┬─────────┘
         │  (CALLER + confident intent only)
         ▼
┌──────────────────┐
│  Qwen 2.5-1.5B   │  generates agent reply in caller's language
└────────┬─────────┘
         │
         ▼
  Dashboard WebSocket  (JSON broadcast to all connected clients)
```

---

## 📁 Project Structure

```
realtime-call-intelligence/
├── src/
│   ├── server.py              # Main WebSocket server
│   └── start_tunnel.py        # ngrok / cloudflared tunnel launcher
├── models/
│   └── train_intent_model.py  # Trains and saves the intent classifier
├── results/
│   └── sample_call_transcript.json   # Example output format
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (T4 or better) for real-time performance.  
CPU works but is too slow for live calls (~10–15× slower).

### 2. Train the intent model

```bash
python models/train_intent_model.py
```

Downloads the MiniLM encoder (~117MB on first run), trains a LogisticRegression classifier on ~700 multilingual examples, and saves `call_intent_model.pkl`.

Expected output:
```
📊 5-fold CV accuracy: 89.2% ± 1.4%
✅ Model saved → call_intent_model.pkl  (246.3 MB)
```

### 3. Start the server

```bash
python src/server.py
```

### 4. Expose it publicly (for telephony provider)

**Using ngrok:**
```bash
python src/start_tunnel.py --tunnel ngrok --token <YOUR_NGROK_AUTH_TOKEN>
```

**Using cloudflared (no account needed):**
```bash
python src/start_tunnel.py --tunnel cloudflared
```

### 5. Configure your telephony provider

Point your provider's streaming URL to:
```
wss://<your-tunnel-url>/
```

Connect your dashboard to:
```
wss://<your-tunnel-url>/dashboard
```

---

## 🧠 Intent Classes

| Intent | Description | Example |
|--------|-------------|---------|
| `order_tracking` | Order/delivery status queries | "mera order kaha hai" |
| `vehicle_inquiry` | Vehicle number or model-specific parts | "my vehicle number is MH12AB1234" |
| `order_return` | Return, refund, or exchange | "wrong part delivered, want refund" |
| `invoice_request` | GST bill / receipt / invoice | "invoice bhej do" |
| `part_availability` | Stock check queries | "brake pad available hai kya" |
| `price_inquiry` | Pricing, discounts, quotations | "clutch plate ka price kya hai" |
| `payment_issue` | Payment failures, double charges | "payment kat gaya order nahi hua" |
| `speak_to_agent` | Escalation to human | "agent se baat karni hai" |
| `other` | Greetings, fillers, non-intent | "okay thanks", "haan", "hello" |

---

## 📊 Example Output

Each utterance produces a JSON event broadcast to the dashboard:

```json
{
  "type": "transcript",
  "speaker": "CALLER",
  "lang": "hi",
  "text": "mera order number hai 78432 aur meri gaadi ka number MH12AB1234 hai",
  "stt_ms": 455,
  "intent": "order_tracking",
  "confidence": 0.89,
  "vehicle_no": "MH12AB1234",
  "order_id": "78432",
  "suggestion": "Order #78432 aapki gaadi MH12AB1234 ke liye hai — main abhi status check kar raha hoon.",
  "qwen_ms": 1920
}
```

See [`results/sample_call_transcript.json`](results/sample_call_transcript.json) for a full example conversation.

---

## ⚙️ Configuration

Key constants at the top of `src/server.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_THRESHOLD` | `0.5` | Silero VAD confidence cutoff |
| `SILENCE_LIMIT` | `20` | Frames of silence before utterance is finalized |
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine sim threshold for speaker matching |
| `INTENT_CONF_THRESHOLD` | `0.60` | Min intent confidence to trigger LLM |
| `QWEN_SYSTEM` | (string) | System prompt — customise for your domain |
| `SPEAKER_LABEL` | `{"SPEAKER_00": "AGENT", "SPEAKER_01": "CALLER"}` | Maps speaker IDs to display labels |

Set `INTENT_MODEL_PATH` env var to change where the model is loaded from:
```bash
export INTENT_MODEL_PATH=/path/to/call_intent_model.pkl
```

---

## 🌐 Telephony Provider Compatibility

The WebSocket protocol expects JSON messages with `event: "start" | "media" | "stop"` and base64-encoded µ-law PCM audio at 8kHz in `media.payload`.

**Tested with:** Exotel (India)  
**Compatible with:** Twilio Media Streams (same format), Vonage (with minor adapter)

---

## 📦 Model Sizes

| Model | Size | Notes |
|-------|------|-------|
| Whisper large-v3 | ~1.5GB | `faster-whisper` int8 quantized |
| Qwen 2.5-1.5B-Instruct | ~3GB | float16 |
| Resemblyzer | ~17MB | d-vector speaker encoder |
| Silero VAD | ~2MB | |
| Intent model (MiniLM + LR) | ~246MB | Saved as `.pkl` |

**Total VRAM required:** ~5GB (fits comfortably on a T4 GPU)

---

## 🔧 Adapting to Your Domain

1. **Change the intent classes** — edit `INTENT_EXAMPLES` in `models/train_intent_model.py` and retrain
2. **Change the system prompt** — edit `QWEN_SYSTEM` in `src/server.py`
3. **Change entity patterns** — edit the regex constants and `PART_KEYWORDS` list in `src/server.py`
4. **Change the LLM** — swap `QWEN_MODEL` for any HuggingFace causal LM

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — Speaker encoder
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — Instruction-tuned LLM
- [sentence-transformers](https://www.sbert.net/) — Multilingual MiniLM encoder
