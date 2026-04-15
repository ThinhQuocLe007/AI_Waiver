# Ollama CLI & GPU Tutorial
> Run LLMs locally from your terminal. No cloud, no API keys, no cost.

---

## 1. What is Ollama?

Ollama is a CLI tool that downloads and runs LLMs locally on your machine using your GPU (or CPU as fallback).

```
Terminal  →  ollama run llama3  →  GPU (NVIDIA/AMD)  →  Response
```

---

## 2. Installation

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### How it runs

The install script creates a **systemd service** that starts automatically. You do **NOT** need to run `ollama serve` manually.

```bash
# Check service status
systemctl status ollama

# Control the service
sudo systemctl start ollama
sudo systemctl stop ollama
sudo systemctl restart ollama

# View live logs
journalctl -u ollama -f
```

> **Common error:** `Error: listen tcp 127.0.0.1:11434: bind: address already in use`
> This means the service is already running. Just use `ollama` commands directly.

---

## 3. Model Management

### Pull (download) a model
```bash
ollama pull llama3              # Llama 3 8B (~4.7 GB)
ollama pull llama3:70b          # Llama 3 70B (~40 GB, needs big GPU)
ollama pull mistral             # Mistral 7B (~4.1 GB)
ollama pull gemma2:9b           # Google Gemma 2 9B
ollama pull qwen2.5:7b          # Qwen 2.5 7B (good for Asian languages)
ollama pull nomic-embed-text    # Embedding model for RAG (~274 MB)
ollama pull phi3:mini           # Microsoft Phi-3 mini (~2.3 GB, lightweight)
```

### List downloaded models
```bash
ollama list
```
```
NAME                ID              SIZE      MODIFIED
llama3:latest       365c0bd3c000    4.7 GB    2 hours ago
nomic-embed-text    0a109f422b47    274 MB    1 hour ago
```

### Show model details
```bash
ollama show llama3
ollama show llama3 --modelfile    # show the full Modelfile
```

### Remove a model
```bash
ollama rm mistral
```

### Copy / rename a model
```bash
ollama cp llama3 my-llama
```

---

## 4. Running Models (Interactive Chat)

### Basic chat
```bash
ollama run llama3
```
- Type your message, press **Enter**
- Multi-line: use `"""` to start/end a block
- Exit: type `/bye`

### With a system prompt
```bash
ollama run llama3 --system "Bạn là nhân viên phục vụ nhà hàng. Luôn trả lời bằng tiếng Việt."
```

### One-shot (non-interactive)
```bash
echo "What is phở?" | ollama run llama3

# Or with a file
ollama run llama3 < question.txt
```

### Adjust generation parameters inline
```bash
ollama run llama3 --verbose    # show timing stats after each response
```

---

## 5. Custom Models (Modelfile)

Create a tailored model with a baked-in system prompt and parameters.

### Create the Modelfile
```bash
cat > Modelfile << 'EOF'
FROM llama3

SYSTEM """
Bạn là nhân viên phục vụ nhà hàng AI Waiter.
Luôn trả lời bằng tiếng Việt.
Giúp khách gọi món, hỏi thực đơn, và thanh toán.
"""

PARAMETER temperature 0.3
PARAMETER num_predict 512
PARAMETER top_p 0.9
EOF
```

### Build & use it
```bash
ollama create ai-waiter -f Modelfile
ollama run ai-waiter
```

### Available PARAMETER options
| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.8 | Creativity (0 = deterministic, 1 = creative) |
| `num_predict` | 128 | Max tokens to generate |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 40 | Top-K sampling |
| `repeat_penalty` | 1.1 | Penalize repeated tokens |
| `num_ctx` | 2048 | Context window size |
| `stop` | — | Stop sequences (e.g. `PARAMETER stop "<|end|>"`) |

---

## 6. REST API (curl)

Ollama exposes a local API at `http://localhost:11434`. Useful for testing before integrating with code.

### Health check
```bash
curl http://localhost:11434
# → "Ollama is running"
```

### Generate (single prompt)
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Explain phở in 2 sentences.",
  "stream": false
}'
```

### Chat (multi-turn)
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    {"role": "system", "content": "You are a Vietnamese waiter."},
    {"role": "user", "content": "Cho tôi xem thực đơn"}
  ],
  "stream": false
}'
```

### List models via API
```bash
curl http://localhost:11434/api/tags
```

### Show running models
```bash
curl http://localhost:11434/api/ps
```

---

## 7. GPU Configuration ⚡

### Check GPU detection
```bash
# What Ollama sees
ollama ps

# NVIDIA GPU stats
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### GPU environment variables

Set these in `/etc/systemd/system/ollama.service` or export before running:

```bash
# Select specific GPU (if you have multiple)
CUDA_VISIBLE_DEVICES=0

# Force CPU only (for testing)
CUDA_VISIBLE_DEVICES=""

# Change listen address (expose to network)
OLLAMA_HOST=0.0.0.0:11434

# Change model storage location
OLLAMA_MODELS=/data/ollama/models

# Set max VRAM to use (in bytes)
OLLAMA_MAX_VRAM=8589934592    # 8 GB
```

### Apply env vars permanently (systemd)
```bash
sudo systemctl edit ollama
```
Add:
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_MAX_VRAM=8589934592"
```
Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### GPU memory usage per model size

| Model | Params | VRAM Required | Speed |
|---|---|---|---|
| phi3:mini | 3.8B | ~3 GB | Very fast |
| llama3:8b | 8B | ~5 GB | Fast |
| mistral:7b | 7B | ~5 GB | Fast |
| gemma2:9b | 9B | ~6 GB | Moderate |
| llama3:70b | 70B | ~40 GB | Slow |

> **Your NVIDIA GPU:** Run `nvidia-smi` to check your available VRAM. For the AI Waiter project, `llama3:8b` (5 GB VRAM) is the sweet spot.

### Quantization (reduce VRAM usage)

Models come in different quantization levels:
```bash
ollama pull llama3:8b-q4_0     # 4-bit quantized, smallest
ollama pull llama3:8b-q5_1     # 5-bit, good balance
ollama pull llama3:8b-fp16     # full precision, biggest
```

| Quantization | VRAM | Quality | Speed |
|---|---|---|---|
| q4_0 | ~4 GB | Good enough | Fastest |
| q5_1 | ~5 GB | Better | Fast |
| q8_0 | ~8 GB | Near perfect | Moderate |
| fp16 | ~16 GB | Best | Slowest |

---

## 8. Multiple Models & Concurrency

```bash
# See what's currently loaded in memory
ollama ps

# Ollama auto-loads/unloads models based on VRAM
# Running a new model unloads the previous one if VRAM is full

# Keep a model loaded (prevents unloading)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "keep_alive": "24h"
}'

# Unload a model immediately
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "keep_alive": 0
}'
```

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `address already in use` | Service is running. Don't use `ollama serve`. Just use CLI. |
| `connection refused` | `sudo systemctl start ollama` |
| Model too slow | Check `nvidia-smi` — is GPU being used? |
| Out of memory (OOM) | Use quantized model (`q4_0`) or close other GPU apps |
| Model not found | `ollama pull MODEL_NAME` first |
| GPU not detected | Check NVIDIA drivers: `nvidia-smi`. Restart: `sudo systemctl restart ollama` |
| Slow first response | Normal — model is loading into VRAM. Subsequent responses are fast. |

### Useful debug commands
```bash
# Service logs
journalctl -u ollama -f

# Check NVIDIA driver
nvidia-smi

# Check Ollama config
ollama show llama3 --modelfile

# Test API directly
curl http://localhost:11434
```

---

## 10. Cheatsheet

| Action | Command |
|---|---|
| **Install** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **Pull model** | `ollama pull llama3` |
| **Run chat** | `ollama run llama3` |
| **Run with system prompt** | `ollama run llama3 --system "prompt"` |
| **List models** | `ollama list` |
| **Running models** | `ollama ps` |
| **Model info** | `ollama show llama3` |
| **Remove model** | `ollama rm llama3` |
| **Copy model** | `ollama cp llama3 my-model` |
| **Create custom** | `ollama create NAME -f Modelfile` |
| **Service status** | `systemctl status ollama` |
| **Restart service** | `sudo systemctl restart ollama` |
| **View logs** | `journalctl -u ollama -f` |
| **API health** | `curl http://localhost:11434` |
| **GPU status** | `nvidia-smi` |
| **Watch GPU** | `watch -n 1 nvidia-smi` |
| **Exit chat** | `/bye` |
