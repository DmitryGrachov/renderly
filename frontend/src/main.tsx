import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { SceneProvider } from './lib/scene'
import { AuthProvider } from './lib/auth'
import { ThemeProvider } from './lib/theme'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <SceneProvider>
          <App />
        </SceneProvider>
      </AuthProvider>
    </ThemeProvider>
  </StrictMode>,
)
