// server.js
// Hugging Face text-to-video microservice.
// Routes video generation through the HF router (fal-ai async-queue protocol),
// so your purchased Hugging Face credits apply.
//
// Designed to run both as a long-running Node server (Railway, VPS)
// and as a completely stateless Serverless Function (Vercel, AWS Lambda).
//
// Endpoints:
//   GET  /health                 liveness + config
//   POST /generate               synchronous: returns video/mp4 bytes
//   POST /jobs                   stateless async: returns a job id immediately
//   GET  /jobs/:id               stateless async: poll job status from HF/fal queue
//   GET  /jobs/:id/video         stateless async: stream finished mp4 or redirect

import express from 'express';
import { InferenceClient } from '@huggingface/inference';

try {
  process.loadEnvFile();
} catch {
  // .env is optional; in cloud environments (Vercel, Railway), env vars are injected directly
}

const {
  HF_TOKEN,
  HF_VIDEO_MODEL = 'Lightricks/LTX-Video-0.9.5',
  HF_PROVIDER = 'fal-ai',
  API_KEY,
  PORT = 3000,
} = process.env;

if (!HF_TOKEN) {
  console.error('FATAL: HF_TOKEN is required. Set it in your environment variables.');
  if (!process.env.VERCEL) {
    process.exit(1);
  }
}

if (!API_KEY) {
  console.warn('WARNING: API_KEY is not set. The service is publicly callable.');
}

const MAX_PROMPT_CHARS = 2000;
const MAX_WAIT_MS = 15 * 60 * 1000;

// Known mappings to avoid extra network lookups
const KNOWN_PROVIDER_PATHS = {
  'Lightricks/LTX-Video-0.9.5': 'fal-ai/ltx-video-v095',
  'Lightricks/LTX-Video-0.9.7-distilled': 'fal-ai/ltx-video-13b-distilled',
  'Lightricks/LTX-Video-0.9.7-dev': 'fal-ai/ltx-video-13b-dev',
  'Wan-AI/Wan2.1-T2V-14B': 'fal-ai/wan-t2v',
  'Wan-AI/Wan2.2-T2V-A14B': 'fal-ai/wan/v2.2-a14b/text-to-video',
  'tencent/HunyuanVideo': 'fal-ai/hunyuan-video',
  'zai-org/CogVideoX-5b': 'fal-ai/cogvideox-5b',
  'genmo/mochi-1-preview': 'fal-ai/mochi-v1',
};

// ---------------------------------------------------------------------------
// HF Inference Client (for synchronous POST /generate)
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

async function resolveProviderPath(model) {
  if (KNOWN_PROVIDER_PATHS[model]) {
    return KNOWN_PROVIDER_PATHS[model];
  }
  if (model.startsWith('fal-ai/')) {
    return model;
  }
  try {
    const res = await fetch(`https://huggingface.co/api/models/${model}?expand[]=inferenceProviderMapping`);
    if (res.ok) {
      const data = await res.json();
      const mapped = data.inferenceProviderMapping?.[HF_PROVIDER]?.providerId;
      if (mapped) {
        KNOWN_PROVIDER_PATHS[model] = mapped;
        return mapped;
      }
    }
  } catch {}
  return model;
}

function encodeJobId(partnerPath, requestId) {
  return Buffer.from(`${partnerPath}:${requestId}`).toString('base64url');
}

function parseJobId(id) {
  try {
    const raw = Buffer.from(id, 'base64url').toString('utf8');
    if (raw.includes(':')) {
      const idx = raw.indexOf(':');
      return {
        partnerPath: raw.slice(0, idx),
        requestId: raw.slice(idx + 1),
      };
    }
  } catch {}
  return {
    partnerPath: KNOWN_PROVIDER_PATHS[HF_VIDEO_MODEL] || 'fal-ai/ltx-video-v095',
    requestId: id,
  };
}

function buildPayload(params) {
  const payload = { prompt: params.prompt };
  if (params.numFrames != null) payload.num_frames = params.numFrames;
  if (params.numInferenceSteps != null) payload.num_inference_steps = params.numInferenceSteps;
  if (params.guidanceScale != null) payload.guidance_scale = params.guidanceScale;
  if (params.negativePrompt != null) payload.negative_prompt = params.negativePrompt;
  if (params.seed != null) payload.seed = params.seed;
  return payload;
}

/**
 * Synchronous generation using HF Inference SDK.
 */
async function generateVideo(params) {
  const model = params.model || HF_VIDEO_MODEL;
  const request = {
    model,
    inputs: params.prompt,
    provider: HF_PROVIDER,
  };

  const parameters = {};
  if (params.numFrames != null) parameters.num_frames = params.numFrames;
  if (params.numInferenceSteps != null) parameters.num_inference_steps = params.numInferenceSteps;
  if (params.guidanceScale != null) parameters.guidance_scale = params.guidanceScale;
  if (params.negativePrompt != null) parameters.negative_prompt = params.negativePrompt;
  if (params.seed != null) parameters.seed = params.seed;

  if (Object.keys(parameters).length > 0) {
    request.parameters = parameters;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MAX_WAIT_MS);

  try {
    const videoBlob = await hf.textToVideo(request, {
      signal: controller.signal,
    });
    const arrayBuffer = await videoBlob.arrayBuffer();
    return { buffer: Buffer.from(arrayBuffer) };
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new HttpError(504, `Timed out after ${MAX_WAIT_MS / 1000}s waiting for video generation`);
    }
    throw new HttpError(
      err.status || 502,
      `Video generation failed: ${err.message}`
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// App Setup
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json({ limit: '256kb' }));

// URL normalization for Vercel and serverless proxies
app.use((req, _res, next) => {
  // 1. If Vercel rewrote the path, restore the original incoming path from header
  const matched = req.headers['x-matched-path'];
  if (matched && typeof matched === 'string' && !matched.endsWith('.js')) {
    req.url = matched;
  }
  // 2. Strip /server.js, /api/index.js, or /api prefixes if present
  if (req.url.startsWith('/server.js/')) {
    req.url = req.url.slice('/server.js'.length);
  } else if (req.url.startsWith('/api/index.js/')) {
    req.url = req.url.slice('/api/index.js'.length);
  } else if (req.url.startsWith('/api/')) {
    req.url = req.url.slice('/api'.length);
  }
  // 3. Fallback: if req.url is literally '/server.js' or '/api/index.js' or '/api'
  if (req.url === '/server.js' || req.url === '/api/index.js' || req.url === '/api') {
    if (req.method === 'POST') {
      req.url = '/jobs';
    } else {
      req.url = '/';
    }
  }
  next();
});

// Simple request log
app.use((req, res, next) => {
  const startedAt = Date.now();
  res.on('finish', () => {
    console.log(`${req.method} ${req.path} -> ${res.statusCode} (${Date.now() - startedAt}ms)`);
  });
  next();
});

// Shared-secret auth. /health stays open for liveness probes.
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
    mode: process.env.VERCEL ? 'serverless' : 'server',
    endpoints: ['GET /health', 'POST /generate', 'POST /jobs', 'GET /jobs/:id', 'GET /jobs/:id/video'],
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    default_model: HF_VIDEO_MODEL,
    provider: HF_PROVIDER,
    auth_required: Boolean(API_KEY),
    serverless: Boolean(process.env.VERCEL),
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

// --- Stateless Asynchronous jobs (100% Serverless & Vercel compatible) ------

app.post('/jobs', async (req, res) => {
  const { prompt, model } = req.body ?? {};
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'prompt (string) is required' });
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    return res.status(400).json({ error: `prompt exceeds ${MAX_PROMPT_CHARS} characters` });
  }

  try {
    const targetModel = model || HF_VIDEO_MODEL;
    const partnerPath = await resolveProviderPath(targetModel);
    const submitUrl = `https://router.huggingface.co/fal-ai/${partnerPath}?_subdomain=queue`;

    const response = await fetch(submitUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(buildPayload(req.body)),
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: `HF submission failed: ${errText}` });
    }

    const data = await response.json();
    const requestId = data.request_id;
    if (!requestId) {
      return res.status(502).json({ error: 'No request_id returned by queue provider' });
    }

    const jobId = encodeJobId(partnerPath, requestId);

    res.status(202).json({
      job_id: jobId,
      status: 'QUEUED',
      status_url: `/jobs/${jobId}`,
      video_url: `/jobs/${jobId}/video`,
    });
  } catch (err) {
    sendError(res, err);
  }
});

app.get('/jobs/:id', async (req, res) => {
  try {
    const { partnerPath, requestId } = parseJobId(req.params.id);
    const statusUrl = `https://router.huggingface.co/fal-ai/${partnerPath}/requests/${requestId}/status?_subdomain=queue`;

    const response = await fetch(statusUrl, {
      headers: { 'Authorization': `Bearer ${HF_TOKEN}` },
    });

    if (response.status === 404) {
      return res.status(404).json({ error: 'job not found' });
    }

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: `Failed to check status: ${errText}` });
    }

    const data = await response.json();
    let status = 'QUEUED';
    if (data.status === 'COMPLETED') status = 'COMPLETED';
    else if (data.status === 'IN_PROGRESS') status = 'IN_PROGRESS';
    else if (data.status === 'FAILED') status = 'FAILED';

    res.json({
      job_id: req.params.id,
      status,
      error: data.error || null,
      queue_position: data.queue_position ?? null,
      metrics: data.metrics || null,
    });
  } catch (err) {
    sendError(res, err);
  }
});

app.get('/jobs/:id/video', async (req, res) => {
  try {
    const { partnerPath, requestId } = parseJobId(req.params.id);

    // 1. Verify status first
    const statusUrl = `https://router.huggingface.co/fal-ai/${partnerPath}/requests/${requestId}/status?_subdomain=queue`;
    const statusRes = await fetch(statusUrl, {
      headers: { 'Authorization': `Bearer ${HF_TOKEN}` },
    });

    if (statusRes.status === 404) {
      return res.status(404).json({ error: 'job not found' });
    }

    const statusData = await statusRes.json();
    if (statusData.status !== 'COMPLETED') {
      const currentStatus = statusData.status === 'IN_PROGRESS' ? 'IN_PROGRESS' : 'QUEUED';
      return res.status(409).json({
        error: `job is ${currentStatus}`,
        ready: false,
        status: currentStatus,
      });
    }

    // 2. Fetch the result
    const resultUrl = `https://router.huggingface.co/fal-ai/${partnerPath}/requests/${requestId}?_subdomain=queue`;
    const resultRes = await fetch(resultUrl, {
      headers: { 'Authorization': `Bearer ${HF_TOKEN}` },
    });

    if (!resultRes.ok) {
      const errText = await resultRes.text();
      return res.status(resultRes.status).json({ error: `Failed to fetch result: ${errText}` });
    }

    const result = await resultRes.json();
    const videoUrl = result.video?.url;
    if (!videoUrl) {
      return res.status(502).json({ error: 'No video URL in completed result', details: result });
    }

    // Support ?redirect=true to directly hand off CDN URL
    if (req.query.redirect === 'true') {
      return res.redirect(302, videoUrl);
    }

    // Otherwise download and stream video bytes
    const videoStream = await fetch(videoUrl);
    if (!videoStream.ok) {
      return res.status(502).json({ error: `Failed to fetch video file from CDN: ${videoStream.statusText}` });
    }

    const buffer = Buffer.from(await videoStream.arrayBuffer());
    res.set({
      'Content-Type': 'video/mp4',
      'Content-Length': String(buffer.length),
      'X-Video-Url': videoUrl,
    });
    res.send(buffer);
  } catch (err) {
    sendError(res, err);
  }
});

// ---------------------------------------------------------------------------
// Error Handler
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

// If running directly as a standalone process (local dev, Railway, Docker)
if (!process.env.VERCEL) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`hf-video listening on 0.0.0.0:${PORT} (model: ${HF_VIDEO_MODEL}, provider: ${HF_PROVIDER})`);
  });
}

// Export for Vercel serverless functions
export default app;