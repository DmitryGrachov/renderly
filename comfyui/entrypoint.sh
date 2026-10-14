#!/usr/bin/env bash
# Докачивает недостающие модели в volume и запускает ComfyUI.
set -euo pipefail
M=/opt/ComfyUI/models

fetch() {  # fetch <папка> <файл> <url> [доп. аргументы curl]
  local dst="$M/$1/$2"
  [ -s "$dst" ] && return 0
  mkdir -p "$M/$1"
  echo "[models] скачиваю $2 ..."
  # mv только после успешной загрузки — иначе пустой файл «прикинулся» бы моделью
  # --retry-all-errors: обрыв посреди файла (curl 18) тоже повторяется, -C - докачивает с места обрыва
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$dst.part" "${@:4}" "$3" && mv "$dst.part" "$dst"
}

HF=https://huggingface.co
fetch diffusion_models flux-2-klein-4b-fp8.safetensors "$HF/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors"
fetch text_encoders    qwen_3_4b.safetensors           "$HF/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
fetch vae              flux2-vae.safetensors           "$HF/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"

# FLUX.2 klein 9B — gated (FLUX Non-Commercial License): примите условия на
# huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 и задайте HF_TOKEN
if [ -n "${HF_TOKEN:-}" ]; then
  fetch diffusion_models flux-2-klein-9b-fp8.safetensors "$HF/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" \
    -H "Authorization: Bearer $HF_TOKEN" \
    || echo "[models] klein 9B не скачана — проверьте HF_TOKEN и принятие лицензии"
  fetch text_encoders    qwen_3_8b_fp8mixed.safetensors  "$HF/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
    || echo "[models] qwen_3_8b не скачан"
else
  echo "[models] HF_TOKEN не задан — FLUX.2 klein 9B пропущена"
fi

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
