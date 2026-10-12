import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // бэкенд из docker compose (BACKEND_PORT) или запущенный вручную
      '/api': process.env.API_PROXY ?? 'http://127.0.0.1:8010',
    },
  },
})
