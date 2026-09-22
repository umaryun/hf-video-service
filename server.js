// server.js
// Hugging Face text-to-video microservice.
// Routes video generation through the HF router (fal async-queue protocol),
// so your purchased Hugging Face credits apply.
//
// Endpoints:
//   GET  /health                 liveness + config
//   POST /generate               synchronous: returns video/mp4 bytes
//   POST /jobs                   async: returns a job id immediately
//   GET  /jobs/:id               async: poll job status
//   GET  /jobs/:id/video         async: fetch the finished mp4

import express from 'express';
import crypto from 'node:crypto';

const {
  HF_TOKEN,
  HF_VIDEO_PATH = 'tencent/HunyuanVideo',
  API_KEY,
  PORT = 3000,
} = process.env;

if (!HF_TOKEN) {
  console.error('FATAL: HF_TOKEN is required. Set it in Railway -> Variables.');
  process.exit(1);
}

if (!API_KEY) {
  console.warn('WARNING: API_KEY is not set. The service is publicly callable.');
}

const ROUTER = 'https://router.huggingface.co/fal-ai';
const POLL_INTERVAL_MS = 5000;
const MAX_WAIT_MS = 15 * 60 * 1000;
const MAX_PROMPT_CHARS = 2000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class HttpError extends Error {
  constructor(status, message, extra = {}) {
    super(message);
    this.status = status;
    this.extra = extra;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * fal returns queue.fal.run URLs. We discard the host and rebuild the path
 * against the HF router - otherwise the Authorization header is rejected.
 * Also strips a trailing /response, since fal returns both shapes:
 *   .../requests/<id>            (as mocked in huggingface_hub tests)
 *   .../requests/<id>/response   (as shown in fal's live docs)
 */
function jobPathFromResponseUrl(responseUrl) {
  if (typeof responseUrl !== 'string') {
    throw new HttpError(502, 'fal response is missing response_url');
  }
  const pathname = new URL(responseUrl).pathname;
  const suffix = '/response';
  return pathname.endsWith(suffix) ? pathname.slice(0, -suffix.length) : pathname;
}

/** Build the fal payload, omitting anything the caller did not supply. */
function buildPayload({ prompt, numFrames, numInferenceSteps, guidanceScale, negativePrompt, seed }) {
  const payload = { prompt };
  if (numFrames != null) payload.num_frames = numFrames;
  if (numInferenceSteps != null) payload.num_inference_steps = numInferenceSteps;
  if (guidanceScale != null) payload.guidance_scale = guidanceScale;
  if (negativePrompt != null) payload.negative_prompt = negativePrompt;
  if (seed != null) payload.seed = seed;
  return payload;
}

/**
 * Full generate cycle: submit -> poll -> fetch result -> download bytes.
 * @param {object} params  prompt, optional model + generation params
 * @param {(status: string) => void} [onProgress]
 */
async function generateVideo(params, onProgress) {
  const auth = { Authorization: `Bearer ${HF_TOKEN}` };
  const modelPath = params.model || HF_VIDEO_PATH;

  // --- 1. Submit to the queue -------------------------------------------
  // ?_subdomain=queue is what routes through fal's async queue and what
  // makes HF credits apply instead of being billed directly by fal.
  const submitRes = await fetch(`${ROUTER}/${modelPath}?_subdomain=queue`, {
    method: 'POST',
    headers: { ...auth, 'Content-Type': 'application/json' },
    body: JSON.stringify(buildPayload(params)),
  });

  const submitText = await submitRes.text();
  if (!submitRes.ok) {
    throw new HttpError(502, `Submit failed [${submitRes.status}]: ${submitText.slice(0, 800)}`);
  }

  let submitted;
  try {
    submitted = JSON.parse(submitText);
  } catch {
    throw new HttpError(502, `Submit returned non-JSON: ${submitText.slice(0, 800)}`);
  }

  if (!submitted.request_id) {
    throw new HttpError(502, `No request_id in response: ${submitText.slice(0, 800)}`);
  }

  const jobPath = jobPathFromResponseUrl(submitted.response_url);
  const statusUrl = `${ROUTER}${jobPath}/status?_subdomain=queue`;
  const resultUrl = `${ROUTER}${jobPath}?_subdomain=queue`;

  // --- 2. Poll until COMPLETED ------------------------------------------
  let status = submitted.status || 'IN_QUEUE';
  const startedAt = Date.now();

  while (status !== 'COMPLETED') {
    if (Date.now() - startedAt > MAX_WAIT_MS) {
      throw new HttpError(504, `Timed out after ${MAX_WAIT_MS / 1000}s. Last status: ${status}`, {
        request_id: submitted.request_id,
      });
    }
    if (status === 'FAILED' || status === 'ERROR') {
      throw new HttpError(502, `Generation failed with status ${status}`, {
        request_id: submitted.request_id,
      });
    }

    await sleep(POLL_INTERVAL_MS);

    const statusRes = await fetch(statusUrl, { headers: auth });
    const statusText = await statusRes.text();
    if (!statusRes.ok) {
      throw new HttpError(502, `Status check failed [${statusRes.status}]: ${statusText.slice(0, 800)}`);
    }

    status = JSON.parse(statusText).status;
    if (onProgress) onProgress(status);
  }

  // --- 3. Fetch the result JSON -----------------------------------------
  const resultRes = await fetch(resultUrl, { headers: auth });
  const resultText = await resultRes.text();
  if (!resultRes.ok) {
    throw new HttpError(502, `Result fetch failed [${resultRes.status}]: ${resultText.slice(0, 800)}`);
  }

  const videoUrl = JSON.parse(resultText)?.video?.url;
  if (!videoUrl) {
    throw new HttpError(502, `No video.url in result: ${resultText.slice(0, 800)}`);
  }

  // --- 4. Download the mp4 bytes (fal's CDN needs no auth) --------------
  const videoRes = await fetch(videoUrl);
  if (!videoRes.ok) {
    throw new HttpError(502, `Video download failed [${videoRes.status}]`);
  }
  const buffer = Buffer.from(await videoRes.arrayBuffer());

  return { buffer, requestId: submitted.request_id, videoUrl };
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json({ limit: '256kb' }));

// Simple request log
app.use((req, res, next) => {
  const startedAt = Date.now();
  res.on('finish', () => {
    console.log(`${req.method} ${req.path} -> ${res.statusCode} (${Date.now() - startedAt}ms)`);
  });
  next();
});

// Shared-secret auth. /health stays open so Railway can healthcheck.
app.use((req, res, next) => {
  if (req.path === '/health') return next();
  if (!API_KEY) return next();
  if (req.get('x-api-key') !== API_KEY) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  next();
});

app.get('/', (_req, res) => {
  res.json({
    service: 'hf-video',
    endpoints: ['GET /health', 'POST /generate', 'POST /jobs', 'GET /jobs/:id', 'GET /jobs/:id/video'],
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    default_model: HF_VIDEO_PATH,
    auth_required: Boolean(API_KEY),
    uptime_s: Math.round(process.uptime()),
  });
});

// --- Synchronous generate ---------------------------------------------------
app.post('/generate', async (req, res) => {
  const { prompt } = req.body ?? {};
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'prompt (string) is required' });
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    return res.status(400).json({ error: `prompt exceeds ${MAX_PROMPT_CHARS} characters` });
  }

  try {
    const { buffer, requestId, videoUrl } = await generateVideo(req.body);
    res.set({
      'Content-Type': 'video/mp4',
      'Content-Length': String(buffer.length),
      'X-Request-Id': requestId,
      'X-Video-Url': videoUrl,
    });
    res.send(buffer);
  } catch (err) {
    sendError(res, err);
  }
});

// --- Asynchronous jobs (recommended for long clips) -------------------------
// NOTE: in-memory store. Jobs are lost on redeploy/restart.
const jobs = new Map();

app.post('/jobs', (req, res) => {
  const { prompt } = req.body ?? {};
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'prompt (string) is required' });
  }

  const id = crypto.randomUUID();
  const job = { id, status: 'QUEUED', progress: null, error: null, createdAt: Date.now() };
  jobs.set(id, job);

  generateVideo(req.body, (status) => {
    job.progress = status;
  })
    .then(({ buffer, requestId, videoUrl }) => {
      Object.assign(job, {
        status: 'COMPLETED',
        buffer,
        requestId,
        videoUrl,
        completedAt: Date.now(),
      });
    })
    .catch((err) => {
      Object.assign(job, {
        status: 'FAILED',
        error: err.message,
        completedAt: Date.now(),
      });
      console.error(`[job ${id}] failed: ${err.message}`);
    });

  res.status(202).json({
    job_id: id,
    status: 'QUEUED',
    status_url: `/jobs/${id}`,
    video_url: `/jobs/${id}/video`,
  });
});

app.get('/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'job not found' });

  res.json({
    job_id: job.id,
    status: job.status,
    progress: job.progress,
    error: job.error,
    video_url: job.videoUrl,
    size_bytes: job.buffer?.length,
  });
});

app.get('/jobs/:id/video', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'job not found' });
  if (job.status !== 'COMPLETED' || !job.buffer) {
    return res.status(409).json({ error: `job is ${job.status}`, ready: false });
  }

  res.set({
    'Content-Type': 'video/mp4',
    'Content-Length': String(job.buffer.length),
    'X-Request-Id': job.requestId,
  });
  res.send(job.buffer);
});

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

function sendError(res, err) {
  const status = err instanceof HttpError ? err.status : 500;
  console.error(`[error] ${err.message}`);
  res.status(status).json({
    error: err.message,
    ...(err instanceof HttpError ? err.extra : {}),
  });
}

process.on('unhandledRejection', (reason) => {
  console.error('[unhandledRejection]', reason);
});

// Railway injects PORT. Must bind 0.0.0.0 to be reachable.
app.listen(PORT, '0.0.0.0', () => {
  console.log(`hf-video listening on 0.0.0.0:${PORT} (default model: ${HF_VIDEO_PATH})`);
});