// server.js
// Hugging Face text-to-video microservice.
// Uses the official @huggingface/inference SDK to route through
// HF Inference Providers (fal-ai), so your HF credits apply.
//
// Endpoints:
//   GET  /health                 liveness + config
//   POST /generate               synchronous: returns video/mp4 bytes
//   POST /jobs                   async: returns a job id immediately
//   GET  /jobs/:id               async: poll job status
//   GET  /jobs/:id/video         async: fetch the finished mp4

import express from 'express';
import crypto from 'node:crypto';
import { InferenceClient } from '@huggingface/inference';

try {
  process.loadEnvFile();
} catch {
  // .env is optional; in production (Railway), env vars are injected directly
}

const {
  HF_TOKEN,
  HF_VIDEO_MODEL = 'Lightricks/LTX-Video-0.9.5',
  HF_PROVIDER = 'fal-ai',
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

const MAX_PROMPT_CHARS = 2000;
const MAX_WAIT_MS = 15 * 60 * 1000;

// ---------------------------------------------------------------------------
// HF Inference Client
// ---------------------------------------------------------------------------

const hf = new InferenceClient(HF_TOKEN);

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

/**
 * Generate a video using the HF Inference SDK.
 * Returns a Buffer of video bytes.
 *
 * @param {object} params  prompt, optional model + generation params
 */
async function generateVideo(params) {
  const model = params.model || HF_VIDEO_MODEL;

  // Build the request for the HF Inference SDK
  const request = {
    model,
    inputs: params.prompt,
    provider: HF_PROVIDER,
  };

  // Add optional parameters if provided
  const parameters = {};
  if (params.numFrames != null) parameters.num_frames = params.numFrames;
  if (params.numInferenceSteps != null) parameters.num_inference_steps = params.numInferenceSteps;
  if (params.guidanceScale != null) parameters.guidance_scale = params.guidanceScale;
  if (params.negativePrompt != null) parameters.negative_prompt = params.negativePrompt;
  if (params.seed != null) parameters.seed = params.seed;

  if (Object.keys(parameters).length > 0) {
    request.parameters = parameters;
  }

  // Use AbortController as an outer timeout safety net
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MAX_WAIT_MS);

  try {
    // textToVideo returns a Blob
    const videoBlob = await hf.textToVideo(request, {
      signal: controller.signal,
    });

    // Convert Blob to Buffer for Express
    const arrayBuffer = await videoBlob.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    return { buffer };
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new HttpError(504, `Timed out after ${MAX_WAIT_MS / 1000}s waiting for video generation`);
    }
    // Re-throw SDK errors with useful context
    throw new HttpError(
      err.status || 502,
      `Video generation failed: ${err.message}`
    );
  } finally {
    clearTimeout(timeout);
  }
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
    default_model: HF_VIDEO_MODEL,
    provider: HF_PROVIDER,
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
    const { buffer } = await generateVideo(req.body);
    res.set({
      'Content-Type': 'video/mp4',
      'Content-Length': String(buffer.length),
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
  const job = { id, status: 'QUEUED', error: null, createdAt: Date.now() };
  jobs.set(id, job);

  generateVideo(req.body)
    .then(({ buffer }) => {
      Object.assign(job, {
        status: 'COMPLETED',
        buffer,
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
    error: job.error,
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
  console.log(`hf-video listening on 0.0.0.0:${PORT} (model: ${HF_VIDEO_MODEL}, provider: ${HF_PROVIDER})`);
});