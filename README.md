# Stable Diffusion 2 Inpainting on Runpod

This repository provides a FastAPI service to run `stabilityai/stable-diffusion-2-inpainting` with endpoints for inpainting and text removal, suitable for deployment on Runpod.

## Project Structure

- Dockerfile
- requirements.txt
- app.py (FastAPI API service)
- start.sh (startup script)
- README.md (this file)

## Local Build & Run

Requirements: Docker with NVIDIA Container Toolkit for GPU access.

```bash
docker build -t sd2-inpaint .
docker run --gpus all -p 7860:7860 sd2-inpaint
```

Service will start at `http://localhost:7860`.

## API

### POST /inpaint

Inputs can be provided as multipart/form-data (recommended) or JSON (base64).

Multipart fields:
- `prompt`: text prompt
- `image`: input image (file or base64 string)
- `mask`: mask image (file or base64 string) — white (255) areas will be inpainted
- Optional params:
  - `negative_prompt` (string)
  - `num_inference_steps` (int, default 30)
  - `guidance_scale` (float, default 7.5)
  - `seed` (int)

JSON fields:
- `prompt`: string
- `image`: base64-encoded PNG/JPEG (optionally with `data:*;base64,` prefix)
- `mask`: base64-encoded PNG/JPEG

Response JSON:
```json
{ "image": "<base64 PNG>" }
```

### Example Request (multipart)

```bash
curl -X POST http://localhost:7860/inpaint \
  -F "prompt=Remove the text" \
  -F "image=@input.png" \
  -F "mask=@mask.png" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=7.5" \
  -F "seed=1234"
```

### Example Request (JSON)

```bash
curl -X POST http://localhost:7860/inpaint \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Remove the text",
    "image": "data:image/png;base64,....",
    "mask": "data:image/png;base64,....",
    "negative_prompt": "text, watermark, letters, characters, logo",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 1234
  }'
### POST /remove_text

Auto-masking helper to remove text and restore background.

Inputs (multipart or JSON):
- `image`: input image
- Optional `mask`: if provided, used as-is; otherwise auto-generated
- Optional params:
  - `prompt` (default: `clean background, remove text`)
  - `negative_prompt` (default: `text, watermark, letters, characters, logo`)
  - `num_inference_steps` (int, default 30)
  - `guidance_scale` (float, default 7.5)
  - `seed` (int)
  - `auto_mask` (bool, default true)
  - `min_text_area` (int, default 100)
  - `dilate_kernel` (int, default 3)

Example (multipart):

```bash
curl -X POST http://localhost:7860/remove_text \
  -F "image=@input.png" \
  -F "auto_mask=true" \
  -F "min_text_area=100" \
  -F "dilate_kernel=3"
```
```

## Health Check

```bash
curl http://localhost:7860/health
```

## Deploy on Runpod

1. Push this project to your GitHub repository.
2. In Runpod, create a new Deployment:
   - Type: Custom Container (Serverless-compatible)
   - Image: Build from this repo (or upload image to a registry)
   - Exposed Port: `7860` (для non-serverless) — для Serverless порт не нужен
   - Command: default (container runs `start.sh`)
   - Runtime: GPU (CUDA-compatible)
3. Set environment variables if needed (optional). `HF_HOME` defaults to `/workspace/huggingface`.
4. For Serverless: set `RUNPOD_SERVERLESS=1`. The container will run `rp_handler.py` and expose a handler via Runpod events API.
5. For non-Serverless: start the pod. The API will be available on the assigned endpoint on port `7860`.

### Notes

- Model is pre-downloaded during the Docker build to speed up startup.
- If running without GPU, the service will use CPU with float32 automatically (slower).
- Build speed tips:
  - Docker cache: keep `requirements.txt` stable to reuse layers.
  - Pre-download HF model during build (already configured).
  - Use `--pull` only when you need latest base image.
 - Serverless input format (Run request body):
   ```json
   {
     "input": {
       "prompt": "Remove the text",
       "image": "<base64 PNG/JPEG>",
       "mask": "<base64 PNG/JPEG>",
       "negative_prompt": "text, watermark, letters, characters, logo",
       "num_inference_steps": 30,
       "guidance_scale": 7.5,
       "seed": 1234,
       "auto_mask": true,
       "min_text_area": 100,
       "dilate_kernel": 3
     }
   }
   ```
   Response:
   ```json
   { "image": "<base64 PNG>" }
   ```


