import base64
import io
import os
from typing import Optional, Tuple

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline


app = FastAPI(title="Stable Diffusion 2 Inpainting API")


MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
HF_CACHE_DIR = os.getenv("HF_HOME")


def _get_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


device, dtype = _get_device_and_dtype()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    cache_dir=HF_CACHE_DIR,
)
pipe = pipe.to(device)


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/inpaint")
async def inpaint(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")

    prompt: Optional[str] = None
    image_data: Optional[bytes] = None
    mask_data: Optional[bytes] = None

    if "application/json" in content_type:
        data = await request.json()
        prompt = data.get("prompt")
        image_field = data.get("image")
        mask_field = data.get("mask")

        if isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)
        if isinstance(mask_field, str):
            mask_data = _b64_to_bytes(mask_field)

    else:
        form = await request.form()
        prompt = form.get("prompt")

        image_field = form.get("image")
        mask_field = form.get("mask")

        # image may be UploadFile or str (base64)
        if isinstance(image_field, UploadFile):
            image_data = await image_field.read()
        elif isinstance(image_field, str):
            image_data = _b64_to_bytes(image_field)

        if isinstance(mask_field, UploadFile):
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

    # Inference
    try:
        use_autocast = device == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            result = pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
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


