#!/usr/bin/env bash
# Докачивает недостающие модели в volume и запускает ComfyUI.
set -euo pipefail
M=/opt/ComfyUI/models

fetch() {  # fetch <папка> <файл> <url>
  local dst="$M/$1/$2"
  [ -s "$dst" ] && return 0
  mkdir -p "$M/$1"
  echo "[models] скачиваю $2 ..."
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dst.part" "$3"
  mv "$dst.part" "$dst"
}

HF=https://huggingface.co
fetch diffusion_models flux-2-klein-4b-fp8.safetensors "$HF/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors"
fetch text_encoders    qwen_3_4b.safetensors           "$HF/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
fetch vae              flux2-vae.safetensors           "$HF/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"

# SegFormer для маски неба/окружения (нод InegoKeepMask)
[ -d "$M/segformer" ] || python -c "
from transformers import SegformerForSemanticSegmentation as M, SegformerImageProcessor as P
i='nvidia/segformer-b4-finetuned-ade-512-512'; P.from_pretrained(i, cache_dir='$M/segformer'); M.from_pretrained(i, cache_dir='$M/segformer')
" || echo "[models] SegFormer не скачан — докачается при первом запросе"

# SAM 2.1 — точный контур главного здания по рамке (только веса transformers, без .pt)
[ -d "$M/sam2" ] || python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam2.1-hiera-large', cache_dir='$M/sam2', allow_patterns=['*.json', 'model.safetensors'])
" || echo "[models] SAM 2.1 не скачан — докачается при первом запросе"

# Depth Anything V2 — делит кадр на фон (позади здания) и окружение (перед ним)
[ -d "$M/depth" ] || python -c "
from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M
i='depth-anything/Depth-Anything-V2-Base-hf'; P.from_pretrained(i, cache_dir='$M/depth'); M.from_pretrained(i, cache_dir='$M/depth')
" || echo "[models] Depth Anything не скачан — докачается при первом запросе"

exec python main.py --listen 0.0.0.0 --port 8188 ${COMFYUI_ARGS:-}
