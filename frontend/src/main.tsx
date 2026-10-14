import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { SceneProvider } from './lib/scene'
import { AuthProvider } from './lib/auth'
import { ThemeProvider } from './lib/theme'
import { AccountProvider } from './lib/account'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <AccountProvider>
          <SceneProvider>
            <App />
          </SceneProvider>
        </AccountProvider>
      </AuthProvider>
    </ThemeProvider>
  </StrictMode>,
)
