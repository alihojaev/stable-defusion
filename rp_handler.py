import base64
import io
import os
from typing import Any, Dict, Tuple

import runpod
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline


MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
HF_CACHE_DIR = os.getenv("HF_HOME")


def _get_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _strip_data_url_prefix(b64: str) -> str:
    if b64.startswith("data:") and ";base64," in b64:
        return b64.split(",", 1)[1]
    return b64


def _b64_to_bytes(b64_str: str) -> bytes:
    b64_clean = _strip_data_url_prefix(b64_str).strip()
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
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, resample=Image.NEAREST)
    return mask_img


device, dtype = _get_device_and_dtype()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    cache_dir=HF_CACHE_DIR,
)
pipe = pipe.to(device)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Runpod Serverless handler.

    Expects event[input] with keys: prompt (str), image (base64), mask (base64).
    Returns: { "image": "<base64 PNG>" } or { "error": "..." }.
    """
    try:
        payload = (event or {}).get("input", {})
        prompt = payload.get("prompt")
        image_field = payload.get("image")
        mask_field = payload.get("mask")

        if not prompt:
            return {"error": "Missing 'prompt'"}
        if not image_field:
            return {"error": "Missing 'image'"}
        if not mask_field:
            return {"error": "Missing 'mask'"}

        image_data = _b64_to_bytes(image_field) if isinstance(image_field, str) else image_field
        mask_data = _b64_to_bytes(mask_field) if isinstance(mask_field, str) else mask_field

        image_pil = _bytes_to_pil_image(image_data, to_rgb=True)
        mask_pil = _bytes_to_pil_image(mask_data, to_rgb=False)
        mask_pil = _ensure_mask(mask_pil, image_pil.size)

        use_autocast = device == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            out = pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
            ).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image": out_b64}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


