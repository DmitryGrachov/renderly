# INEGO.dev

AI-инструменты для архитектурной визуализации поверх ComfyUI.

```
frontend/  React + Vite + TypeScript + Tailwind v4
backend/   FastAPI (заглушка, ComfyUI ещё не подключён)
```

## Docker (рекомендуется)

```bash
docker compose up -d --build     # http://localhost:8080
docker compose logs -f           # логи
docker compose down              # остановить
```

- `frontend` — сборка Vite + nginx, отдаёт статику и проксирует `/api/*` в `backend`.
- `backend` — FastAPI на `:8000` (наружу не публикуется, только через nginx).
- Настройки — в `.env` в корне (см. `.env.example`): порт сайта, мок/реальная генерация, адрес ComfyUI.
  По умолчанию ComfyUI ищется на хосте: `http://host.docker.internal:8188`.
- После изменения `VITE_USE_MOCK` нужна пересборка: `docker compose up -d --build frontend`.

## Frontend (локально, без Docker)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

По умолчанию генерация имитируется в браузере (`VITE_USE_MOCK=true`).
Чтобы ходить в бэкенд, создайте `frontend/.env` с `VITE_USE_MOCK=false` — запросы `/api/*` проксируются на `:8000`.

## Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Адрес ComfyUI задаётся через `COMFYUI_URL` (по умолчанию `http://127.0.0.1:8188`).
"# Renderly" 
