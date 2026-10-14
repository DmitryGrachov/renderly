# INEGO.dev

AI-инструменты для архитектурной визуализации поверх ComfyUI.

```
frontend/  React + Vite + TypeScript + Tailwind v4
backend/   FastAPI + PostgreSQL (авторизация, DeepSeek → промпт, клиент ComfyUI)
comfyui/   ComfyUI + FLUX.2 [klein] 4B / 9B в Docker (GPU)
```

## Docker (рекомендуется)

```bash
cp .env.example .env             # и впишите JWT_SECRET
docker compose up -d --build     # http://localhost:8080
docker compose logs -f           # логи
docker compose down              # остановить (данные БД сохраняются в volume pgdata)
```

- `frontend` — сборка Vite + nginx, отдаёт статику и проксирует `/api/*` в `backend`.
- `backend` — FastAPI. Наружу открыт только на `127.0.0.1:${BACKEND_PORT:-8010}` — для `npm run dev`.
- `db` — PostgreSQL 16, данные в volume `pgdata`. Таблицы создаются при старте бэкенда.
- `comfyui` — ComfyUI на GPU (нужен NVIDIA Container Toolkit / Docker Desktop с WSL2). При первом старте
  докачивает в volume `comfy_models` ~14 ГБ: `flux-2-klein-4b-fp8`, `qwen_3_4b`, `flux2-vae`, SegFormer, SAM 2.1, Depth Anything V2. UI — http://127.0.0.1:8188.
  FLUX.2 klein 9B (`flux-2-klein-9b-fp8` + `qwen_3_8b_fp8mixed`, ещё ~17 ГБ) скачивается, только если задан `HF_TOKEN`:
  репозиторий gated (FLUX Non-Commercial License) — сначала примите условия на странице модели на Hugging Face.
- Настройки — в `.env` в корне (см. `.env.example`).
- После изменения `VITE_USE_MOCK` нужна пересборка: `docker compose up -d --build frontend`.

## Авторизация

- `POST /api/auth/register` `{name, email, password}` · `POST /api/auth/login` `{email, password}` ·
  `POST /api/auth/logout` · `GET /api/auth/me`
- Пароли — Argon2. Сессия — JWT в httpOnly-cookie `inego_session` (SameSite=Lax, срок — `JWT_TTL_DAYS`).
- `/api/img2img` и `/api/inpaint` требуют входа.
- Без `JWT_SECRET` секрет генерируется при каждом старте — все сессии сбрасываются после перезапуска.
- За HTTPS включите `COOKIE_SECURE=true`.

## Frontend (локально)

```bash
docker compose up -d db backend  # бэкенд нужен для входа
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

`/api/*` проксируется на `http://127.0.0.1:8010` (переопределяется переменной `API_PROXY`).
Генерация по умолчанию имитируется в браузере (`VITE_USE_MOCK=true`); вход и регистрация — всегда через бэкенд.

## Backend (локально, без Docker)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set DATABASE_URL=postgresql+asyncpg://inego:inego@localhost:5432/inego
uvicorn app.main:app --reload --port 8010
```

Адрес ComfyUI задаётся через `COMFYUI_URL` (по умолчанию `http://127.0.0.1:8188`).

## Img2img экстерьера (FLUX.2 klein 4B / 9B)

Модель выбирается в запросе: поле `model` в `POST /api/img2img` — `klein-4b` (по умолчанию) или `klein-9b`.
Если веса выбранной модели ещё не скачаны в ComfyUI, бэкенд отвечает 503 с понятной ошибкой.

1. **DeepSeek** (`deepseek-flash`, vision) смотрит на исходник и пишет инструкцию редактирования: материалы,
   окружение и свет пресета + обязательный блок «сохранить камеру, перспективу, геометрию, проёмы».
   Для «Своего референса» DeepSeek сначала описывает свет и колористику загруженного фото.
   Без `DEEPSEEK_API_KEY` или при ошибке API используется шаблонный промпт.
2. **ComfyUI**: исходник → `ReferenceLatent` (image 1) — модель редактирует его, а не рисует заново;
   референс света пресета (`backend/app/refs/exterior/*.jpg`) → второй `ReferenceLatent` (image 2) — только колористика.
   Сэмплинг стартует с латента исходника; «Сила» 10–100% → denoise 0.75–1.0 (8 шагов, euler, cfg 1).
3. «Сохранить небо / фон / окружение» — сохраняется содержимое зоны (те же облака, застройка, дороги, машины),
   а свет и цвет берутся по референсу. Два прохода: полный рендер → нод `InegoKeepZones` размечает зоны
   ([comfyui/custom_nodes/inego_nodes](comfyui/custom_nodes/inego_nodes/__init__.py): небо — SegFormer, главное здание —
   SAM 2.1 по рамке от DeepSeek, деление фон/окружение — по классам и глубине Depth Anything V2) → `InegoZoneRestyle`
   ставит в сохраняемые зоны исходник в цвете рендера (LAB + низкочастотный свет) → лёгкая гармонизация
   (`HARMONIZE`, по умолчанию 2 шага из 8). Фон — прочая застройка и всё позади здания, окружение — земля,
   озеленение, люди, машины перед ним.
4. Кадр ~1.5 Мп (`RENDER_MEGAPIXELS`) в пропорциях исходника; ответ — JPEG в точной пропорции исходника.

Свои референсы пресетов — заменить файлы в `backend/app/refs/exterior/` и описание `look` в `backend/app/presets.py`.
