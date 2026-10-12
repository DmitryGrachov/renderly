# INEGO.dev

AI-инструменты для архитектурной визуализации поверх ComfyUI.

```
frontend/  React + Vite + TypeScript + Tailwind v4
backend/   FastAPI + PostgreSQL (авторизация; ComfyUI ещё не подключён)
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
