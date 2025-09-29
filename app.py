import base64
import io
import os
from typing import Optional, Tuple

from fastapi import FastAPI, Request, UploadFile, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
import cv2
# Lazy import of diffusers inside get_pipeline to avoid heavy import at startup


app = FastAPI(title="Stable Diffusion 2 Inpainting API")


MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
HF_CACHE_DIR = os.getenv("HF_HOME")


def _get_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


device, dtype = _get_device_and_dtype()
_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        from diffusers import StableDiffusionInpaintPipeline  # imported lazily
        pipe_local = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            cache_dir=HF_CACHE_DIR,
        )
        pipe_local = pipe_local.to(device)
        # Memory optimizations
        try:
            pipe_local.enable_attention_slicing()
        except Exception:
            pass
        _pipe = pipe_local
    return _pipe


def _strip_data_url_prefix(b64: str) -> str:
    if b64.startswith("data:") and ";base64," in b64:
        return b64.split(",", 1)[1]
    return b64


def _b64_to_bytes(b64_str: str) -> bytes:
    b64_clean = _strip_data_url_prefix(b64_str).strip()
    # Fix padding if missing
    padding = len(b64_clean) % 4
    if padding:
        b64_clean += "=" * (4 - padding)
    return base64.b64decode(b64_clean)


def _bytes_to_pil_image(data: bytes, to_rgb: bool = True) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if to_rgb and img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _ensure_mask(mask_img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    # Inpainting mask expects white (255) regions to be inpainted
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, resample=Image.NEAREST)
    return mask_img


def _auto_text_mask(
    image: Image.Image,
    min_text_area: int = 100,
    dilate_kernel: int = 3,
) -> Image.Image:
    """Create a heuristic text mask from an RGB PIL image.
    Returns a single-channel (L) PIL image with white regions to inpaint.
    """
    np_img = np.array(image)
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img if np_img.ndim == 2 else cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    dilate_k = max(1, int(dilate_kernel))
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
    dilated = cv2.dilate(opened, dil_kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= int(min_text_area):
            mask[labels == i] = 255

    return Image.fromarray(mask, mode="L")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/inpaint")
async def inpaint(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    image_data: Optional[bytes] = None
    mask_data: Optional[bytes] = None

    if "application/json" in content_type:
        data = await request.json()
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt")
        num_inference_steps = data.get("num_inference_steps")
        guidance_scale = data.get("guidance_scale")
        seed = data.get("seed")
        image_field = data.get("image")
        mask_field = data.get("mask")

        if isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)
        if isinstance(mask_field, str):
            mask_data = _b64_to_bytes(mask_field)

    else:
        form = await request.form()
        prompt = form.get("prompt")
        negative_prompt = form.get("negative_prompt")
        num_inference_steps = form.get("num_inference_steps")
        guidance_scale = form.get("guidance_scale")
        seed = form.get("seed")

        image_field = form.get("image")
        mask_field = form.get("mask")

        # image may be UploadFile (FastAPI/Starlette) or str (base64)
        if isinstance(image_field, (UploadFile, StarletteUploadFile)) or hasattr(image_field, "read"):
            image_data = await image_field.read()
        elif isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)

        if isinstance(mask_field, (UploadFile, StarletteUploadFile)) or hasattr(mask_field, "read"):
            mask_data = await mask_field.read()
        elif isinstance(mask_field, str):
            mask_data = _b64_to_bytes(mask_field)

    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing 'image' (file or base64).")
    if not mask_data:
        raise HTTPException(status_code=400, detail="Missing 'mask' (file or base64).")

    try:
        image_pil = _bytes_to_pil_image(image_data, to_rgb=True)
        mask_pil = _bytes_to_pil_image(mask_data, to_rgb=False)
        mask_pil = _ensure_mask(mask_pil, image_pil.size)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse images: {e}")

    # Defaults
    if num_inference_steps is None:
        num_inference_steps = 30
    else:
        try:
            num_inference_steps = int(num_inference_steps)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'num_inference_steps'")
    if guidance_scale is None:
        guidance_scale = 7.5
    else:
        try:
            guidance_scale = float(guidance_scale)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'guidance_scale'")

    # Inference
    try:
        use_autocast = device == "cuda"
        pipe = get_pipeline()
        generator = None
        if seed is not None:
            try:
                seed_val = int(seed)
                generator = torch.Generator(device=device).manual_seed(seed_val)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'seed'")
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            result = pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({"image": out_b64})


@app.post("/remove_text")
async def remove_text(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    auto_mask: bool = True
    min_text_area: int = 100
    dilate_kernel: int = 3

    image_data: Optional[bytes] = None
    mask_data: Optional[bytes] = None

    if "application/json" in content_type:
        data = await request.json()
        prompt = data.get("prompt") or "clean background, remove text"
        negative_prompt = data.get("negative_prompt") or "text, watermark, letters, characters, logo"
        num_inference_steps = data.get("num_inference_steps")
        guidance_scale = data.get("guidance_scale")
        seed = data.get("seed")
        auto_mask = bool(data.get("auto_mask", True))
        min_text_area = int(data.get("min_text_area", 100))
        dilate_kernel = int(data.get("dilate_kernel", 3))

        image_field = data.get("image")
        mask_field = data.get("mask")
        if isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)
        if isinstance(mask_field, str):
            mask_data = _b64_to_bytes(mask_field)
    else:
        form = await request.form()
        prompt = form.get("prompt") or "clean background, remove text"
        negative_prompt = form.get("negative_prompt") or "text, watermark, letters, characters, logo"
        num_inference_steps = form.get("num_inference_steps")
        guidance_scale = form.get("guidance_scale")
        seed = form.get("seed")
        auto_mask = (form.get("auto_mask") or "true").lower() not in {"false", "0", "no"}
        try:
            min_text_area = int(form.get("min_text_area") or 100)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'min_text_area'")
        try:
            dilate_kernel = int(form.get("dilate_kernel") or 3)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'dilate_kernel'")

        image_field = form.get("image")
        mask_field = form.get("mask")
        if isinstance(image_field, (UploadFile, StarletteUploadFile)) or hasattr(image_field, "read"):
            image_data = await image_field.read()
        elif isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)
        if isinstance(mask_field, (UploadFile, StarletteUploadFile)) or hasattr(mask_field, "read"):
            mask_data = await mask_field.read()
        elif isinstance(mask_field, str):
            mask_data = _b64_to_bytes(mask_field)

    if not image_data:
        raise HTTPException(status_code=400, detail="Missing 'image' (file or base64).")

    try:
        image_pil = _bytes_to_pil_image(image_data, to_rgb=True)
        if mask_data is not None:
            mask_pil = _bytes_to_pil_image(mask_data, to_rgb=False)
            mask_pil = _ensure_mask(mask_pil, image_pil.size)
        elif auto_mask:
            mask_pil = _auto_text_mask(image_pil, min_text_area=min_text_area, dilate_kernel=dilate_kernel)
            mask_pil = _ensure_mask(mask_pil, image_pil.size)
        else:
            raise HTTPException(status_code=400, detail="Missing 'mask'. Provide 'mask' or set 'auto_mask=true'.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse/generate mask: {e}")

    if num_inference_steps is None:
        num_inference_steps = 30
    else:
        try:
            num_inference_steps = int(num_inference_steps)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'num_inference_steps'")
    if guidance_scale is None:
        guidance_scale = 7.5
    else:
        try:
            guidance_scale = float(guidance_scale)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'guidance_scale'")

    try:
        use_autocast = device == "cuda"
        pipe = get_pipeline()
        generator = None
        if seed is not None:
            try:
                seed_val = int(seed)
                generator = torch.Generator(device=device).manual_seed(seed_val)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'seed'")
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            result = pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({"image": out_b64})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


