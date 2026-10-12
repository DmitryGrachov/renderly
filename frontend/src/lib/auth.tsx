import { createContext, useCallback, useContext, useEffect, useState } from 'react'

export interface User {
  id: string
  email: string
  name: string
  created_at: string
}

type Status = 'loading' | 'authed' | 'guest'

const MESSAGES: Record<string, string> = {
  invalid_credentials: 'Неверная почта или пароль',
  email_taken: 'Аккаунт с такой почтой уже существует',
  invalid_email: 'Этот адрес почты не подходит — проверьте домен',
  validation: 'Проверьте правильность заполнения полей',
  server: 'Сервер недоступен. Убедитесь, что бэкенд запущен',
  unknown: 'Что-то пошло не так. Попробуйте ещё раз',
}

export class AuthError extends Error {
  constructor(public code: string) {
    super(MESSAGES[code] ?? MESSAGES.unknown)
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api/auth/${path}`, {
      method: body === undefined && path === 'me' ? 'GET' : 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      credentials: 'same-origin',
    })
  } catch {
    throw new AuthError('server')
  }
  if (res.ok) return (res.status === 204 ? undefined : await res.json()) as T
  if (res.status >= 500) throw new AuthError('server')
  const detail = await res.json().catch(() => null)
  if (res.status === 422) {
    // pydantic: [{ loc: ['body', 'email'], ... }]
    const emailBad = Array.isArray(detail?.detail) && detail.detail.some((d: { loc?: unknown[] }) => d.loc?.includes('email'))
    throw new AuthError(emailBad ? 'invalid_email' : 'validation')
  }
  throw new AuthError(typeof detail?.detail === 'string' ? detail.detail : 'unknown')
}

interface AuthContextValue {
  user: User | null
  status: Status
  login: (email: string, password: string) => Promise<void>
  register: (name: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [status, setStatus] = useState<Status>('loading')

  useEffect(() => {
    request<User>('me')
      .then((u) => { setUser(u); setStatus('authed') })
      .catch(() => setStatus('guest'))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    setUser(await request<User>('login', { email, password }))
    setStatus('authed')
  }, [])

  const register = useCallback(async (name: string, email: string, password: string) => {
    setUser(await request<User>('register', { name, email, password }))
    setStatus('authed')
  }, [])

  const logout = useCallback(async () => {
    await request<void>('logout', {}).catch(() => {})
    setUser(null)
    setStatus('guest')
  }, [])

  return <AuthContext.Provider value={{ user, status, login, register, logout }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
