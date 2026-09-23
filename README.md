# hf-video-service

A lightweight Node.js microservice that acts as a bridge between workflow automation tools (like **n8n**, Make, or custom webhooks) and **Hugging Face Text-to-Video** inference providers (via `fal-ai`).

Using the official `@huggingface/inference` SDK, requests route directly through Hugging Face's router so your **prepaid Hugging Face credits** are used automatically.

---

## Features

- **Cheapest Default Model**: Pre-configured with `Lightricks/LTX-Video-0.9.5` (~**$0.04 per video**, ~40s generation time).
- **Official SDK Integration**: Powered by `@huggingface/inference` for reliability and clean error handling.
- **Dual Operating Modes**:
  - **Synchronous (`POST /generate`)**: Waits for generation and returns binary `video/mp4` directly (great for simple workflows).
  - **Asynchronous (`POST /jobs`)**: Immediately returns a `job_id` for long-running generation, allowing non-blocking polling and video retrieval.
- **Authentication**: Shared-secret API key protection via `x-api-key` header (`/health` remains open for cloud health checks).
- **Zero-Dependency Env Loading**: Uses Node.js native `process.loadEnvFile()` for effortless local development.
- **Customizable Inference**: Pass model overrides, steps, frame count, guidance scale, and seeds dynamically per request.

---

## Quick Start

### 1. Requirements

- Node.js >= 20.6.0
- A Hugging Face account with a user access token (`HF_TOKEN`) and billing/credits enabled.

### 2. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/umaryun/hf-video-service.git
cd hf-video-service
npm install
```

Create a `.env` file in the root directory:

```env
HF_TOKEN="hf_your_token_here"
HF_VIDEO_MODEL="Lightricks/LTX-Video-0.9.5"
HF_PROVIDER="fal-ai"
API_KEY="your-secret-api-key"
PORT=3000
```

### 3. Run Locally

```bash
npm start
```

---

## Configuration (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | — | Hugging Face Access Token with `inference` permission. |
| `HF_VIDEO_MODEL` | No | `Lightricks/LTX-Video-0.9.5` | Default text-to-video model identifier. |
| `HF_PROVIDER` | No | `fal-ai` | Inference provider to route through (e.g. `fal-ai`, `replicate`). |
| `API_KEY` | No | — | Optional secret token. If set, required on all requests via `x-api-key` header. |
| `PORT` | No | `3000` | Port to bind to (Railway automatically injects this). |

---

## Supported Models

You can change the default model via `HF_VIDEO_MODEL` or override it per request in the JSON body:

| Model | Cost / Video | Generation Time | Notes |
|---|---|---|---|
| **`Lightricks/LTX-Video-0.9.5`** *(default)* | **~$0.04** | **~40s** | **Recommended.** Cheapest working model, fast inference, high reliability. |
| `Wan-AI/Wan2.1-T2V-14B` | ~$0.40 | ~85s | Higher parameter count (14B), higher visual fidelity, higher cost. |

---

## API Reference

All protected endpoints require the `x-api-key` header if `API_KEY` is set in the environment.

### 1. Health Check
Public endpoint (does not require `x-api-key`). Useful for Railway / container healthchecks.

```bash
GET /health
```

**Response:**
```json
{
  "ok": true,
  "default_model": "Lightricks/LTX-Video-0.9.5",
  "provider": "fal-ai",
  "auth_required": true,
  "uptime_s": 42
}
```

---

### 2. Synchronous Generation (`POST /generate`)
Generates video and blocks until finished, returning raw `video/mp4` binary bytes.

```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{"prompt": "A majestic eagle soaring over snow-capped mountains at sunrise"}' \
  --output video.mp4
```

**Response:**
- `Content-Type: video/mp4`
- Body: Binary MP4 stream

---

### 3. Asynchronous Job Workflow (Recommended for n8n)

For workflows with timeouts or long-running prompts, use the asynchronous job queue.

#### Step 3a: Submit Job (`POST /jobs`)
```bash
curl -X POST http://localhost:3000/jobs \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "prompt": "Cinematic slow motion of waves crashing against rocky cliffs, 4k",
    "numFrames": 161,
    "seed": 42
  }'
```

**Response (`202 Accepted`):**
```json
{
  "job_id": "6c208fd0-0a20-435a-b38d-e462ca51655f",
  "status": "QUEUED",
  "status_url": "/jobs/6c208fd0-0a20-435a-b38d-e462ca51655f",
  "video_url": "/jobs/6c208fd0-0a20-435a-b38d-e462ca51655f/video"
}
```

#### Step 3b: Poll Status (`GET /jobs/:id`)
```bash
curl http://localhost:3000/jobs/6c208fd0-0a20-435a-b38d-e462ca51655f \
  -H "x-api-key: your-secret-api-key"
```

**Response (In progress):**
```json
{
  "job_id": "6c208fd0-0a20-435a-b38d-e462ca51655f",
  "status": "QUEUED",
  "error": null
}
```

**Response (Completed):**
```json
{
  "job_id": "6c208fd0-0a20-435a-b38d-e462ca51655f",
  "status": "COMPLETED",
  "error": null,
  "size_bytes": 3823322
}
```

#### Step 3c: Retrieve Video (`GET /jobs/:id/video`)
```bash
curl http://localhost:3000/jobs/6c208fd0-0a20-435a-b38d-e462ca51655f/video \
  -H "x-api-key: your-secret-api-key" \
  --output final_video.mp4
```

**Response:**
- `Content-Type: video/mp4`
- Body: Binary MP4 stream

---

## n8n Integration

### Synchronous Flow (3 nodes)
```
[Trigger] → [HTTP Request: POST /generate] → [Google Drive: Upload File]
```
- In the **HTTP Request** node:
  - **Method**: `POST`
  - **URL**: `https://your-service.up.railway.app/generate`
  - **Headers**: `x-api-key: your-api-key`
  - **Response Format**: `File`
  - **Binary Property**: `data`
- Connect directly to the **Google Drive** node set to `Upload a file` using input binary field `data`.

### Asynchronous Flow (with polling)
```
[Trigger] → [POST /jobs] → [Wait 30s] → [GET /jobs/:id] → [If COMPLETED]
                                               ↑                  |
                                               └── (If not) ──────┘
                                                                  ↓
                                                   [GET /jobs/:id/video] → [Google Drive]
```

---

## Deployment (Railway)

1. Push your repository to GitHub.
2. In [Railway](https://railway.app), create a **New Project** and select your GitHub repo.
3. In **Variables**, set:
   - `HF_TOKEN`: Your Hugging Face user token
   - `HF_VIDEO_MODEL`: `Lightricks/LTX-Video-0.9.5`
   - `HF_PROVIDER`: `fal-ai`
   - `API_KEY`: Your custom secret key
4. Railway will automatically inject `PORT` and deploy.
5. In **Settings -> Networking**, click **Generate Domain** to get your public URL.

---

## License

MIT
