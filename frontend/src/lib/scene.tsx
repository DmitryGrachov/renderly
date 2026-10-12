import { createContext, useContext, useEffect, useState } from 'react'

export type Scene = 'interior' | 'exterior'

export const SCENE_LABEL: Record<Scene, string> = {
  interior: 'Интерьер',
  exterior: 'Экстерьер',
}

const STORAGE_KEY = 'inego.scene'

const read = (): Scene => {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'interior' || v === 'exterior') return v
  } catch { /* storage недоступен */ }
  return 'exterior'
}

const SceneContext = createContext<{ scene: Scene; setScene: (s: Scene) => void } | null>(null)

export function SceneProvider({ children }: { children: React.ReactNode }) {
  const [scene, setScene] = useState<Scene>(read)
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, scene) } catch { /* storage недоступен */ }
  }, [scene])
  return <SceneContext.Provider value={{ scene, setScene }}>{children}</SceneContext.Provider>
}

export function useScene() {
  const ctx = useContext(SceneContext)
  if (!ctx) throw new Error('useScene must be used inside SceneProvider')
  return ctx
}
