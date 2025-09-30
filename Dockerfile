FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Hugging Face cache directory and optional auth token for gated models
ARG HF_TOKEN=""
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV HF_HOME=/workspace/huggingface
RUN mkdir -p ${HF_HOME}

# Pre-download the Stable Diffusion 2 Inpainting model to the cache (non-fatal)
RUN python3 -c "import os, torch; from diffusers import StableDiffusionInpaintPipeline as P;\
model_id='stabilityai/stable-diffusion-2-inpainting'; dtype=torch.float16; cache_dir=os.environ.get('HF_HOME');\
print('Pre-downloading model...');\
P.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir);\
print('Model predownload completed.')" || true

# Copy application files
COPY app.py /workspace/app.py
COPY start.sh /workspace/start.sh
COPY rp_handler.py /workspace/rp_handler.py
RUN chmod +x /workspace/start.sh

EXPOSE 7860

CMD ["/workspace/start.sh"]


