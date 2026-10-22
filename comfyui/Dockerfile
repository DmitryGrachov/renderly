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

# перенос образца текстуры (Inpainting «Материалы»): MoGe-2 — геометрия граней в метрах (MIT), Marigold-IID —
# разложение исходника на альбедо и освещённость (diffusers, веса OpenRAIL++-M). --no-deps: torch и numpy не трогаем
RUN pip install --no-deps \
        "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
        "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
        "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
    && pip install opencv-python-headless==5.0.0.93 diffusers==0.41.0 vtracer==0.6.15

# Render pass «кадр по материалам»: код Materialistic (Sharma et al., SIGGRAPH 2023; в репозитории нет лицензии —
# только исследовательское использование). Веса — в volume моделей (entrypoint.sh)
RUN git clone https://github.com/prafull7/materialistic.git /opt/materialistic     && git -C /opt/materialistic checkout bc2a444149decf19ec499f4caf50c7b2837906d6

# Render pass: кадр без света до схемы материалов — Intrinsic v2 (Careaga & Aksoy, TOG 2024; только академическое
# использование) со своими форками MiDaS и chrislib. --no-deps: torch, numpy и pillow не трогаем, остальное — списком.
# Веса докачиваются при первом запросе в volume (models/intrinsic/hub)
RUN pip install --no-deps \
        "intrinsic @ git+https://github.com/compphoto/Intrinsic.git@1a93aafaaf7dbe11be7ec86323d9a06e5950966d" \
        "altered_midas @ git+https://github.com/CCareaga/MiDaS@fb51e3a" \
        "chrislib @ git+https://github.com/CCareaga/chrislib@9a4c63f" \
        timm==0.9.16 scikit-image==0.26.0 lazy_loader==0.6 imageio==2.38.1 tifffile==2026.10.15 \
        matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 fonttools==4.67.0 kiwisolver==1.5.1 pyparsing==3.3.3 \
        python-dateutil==3.0.0 six==1.17.0 beautifulsoup4==4.15.0 soupsieve==3.0.4 antialiased_cnns==0.3

COPY custom_nodes/inego_nodes /opt/ComfyUI/custom_nodes/inego_nodes
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188
HEALTHCHECK --interval=15s --timeout=5s --start-period=30m --retries=5 CMD curl -fs http://127.0.0.1:8188/system_stats || exit 1
ENTRYPOINT ["/entrypoint.sh"]
