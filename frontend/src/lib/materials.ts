/** Render pass: /api/materials — кадр → плоская иллюстрация Qwen-Image-Edit-2511 */

export interface RenderPassResult {
  /** размер кадра (уменьшен до 2048 по длинной стороне) */
  width: number
  height: number
  /** плоская иллюстрация 2511 на месте кадра (PNG data URL) */
  schema: string
}

export async function buildMaterials(file: File, signal?: AbortSignal): Promise<RenderPassResult> {
  const body = new FormData()
  body.append('image', file)
  const res = await fetch('/api/materials', { method: 'POST', body, signal })
  if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
  if (res.status === 413) throw new Error('Файл слишком большой')
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail as string).catch(() => '')
    throw new Error(detail || `Сервер ответил ${res.status}`)
  }
  const j = await res.json()
  return { width: j.width, height: j.height, schema: j.schema }
}
