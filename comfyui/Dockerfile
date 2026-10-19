# ComfyUI + PyTorch (CUDA 12.8 — нужен для RTX 50xx / Blackwell)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG COMFYUI_REF=master
RUN git clone --depth 1 --branch ${COMFYUI_REF} https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && pip install -r /opt/ComfyUI/requirements.txt

# ComfyUI-GGUF — загрузчик сжатых GGUF-моделей (Qwen-Image-Edit-2511 Q4_K_M помещается в 16 ГБ видеопамяти)
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF     && pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt

COPY custom_nodes/inego_nodes /opt/ComfyUI/custom_nodes/inego_nodes
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188
HEALTHCHECK --interval=15s --timeout=5s --start-period=30m --retries=5 CMD curl -fs http://127.0.0.1:8188/system_stats || exit 1
ENTRYPOINT ["/entrypoint.sh"]
