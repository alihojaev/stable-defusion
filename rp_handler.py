import base64
import io
import os
from typing import Any, Dict, Tuple

import runpod
from PIL import Image
import torch
# Lazy import inside get_pipeline


MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
HF_CACHE_DIR = os.getenv("HF_HOME")
ENV_DISABLE_SAFETY = os.getenv("DISABLE_SAFETY_CHECKER", "0") == "1"


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
        try:
            pipe_local.enable_attention_slicing()
        except Exception:
            pass
        if ENV_DISABLE_SAFETY:
            try:
                pipe_local.safety_checker = None
            except Exception:
                pass
        _pipe = pipe_local
    return _pipe


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
        enable_safety_checker = payload.get("enable_safety_checker")

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
        pipe = get_pipeline()
        if enable_safety_checker is not None and not enable_safety_checker:
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        try:
            with torch.autocast(device_type="cuda", enabled=use_autocast):
                out = pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                ).images[0]
        except Exception as e:
            msg = str(e)
            if "no kernel image is available for execution on the device" in msg.lower() or "cuda error" in msg.lower():
                # Fallback to CPU
                try:
                    pipe = None
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                global _pipe, device, dtype
                _pipe = None
                device, dtype = "cpu", torch.float32
                pipe = get_pipeline()
                out = pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                ).images[0]
            else:
                raise

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image": out_b64}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


