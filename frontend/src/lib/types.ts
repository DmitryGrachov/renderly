export interface StudioImage {
  file: File
  url: string
  name: string
  width: number
  height: number
}

export async function toStudioImage(file: File): Promise<StudioImage> {
  const url = URL.createObjectURL(file)
  const { width, height } = await new Promise<{ width: number; height: number }>((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = reject
    img.src = url
  })
  return { file, url, name: file.name, width, height }
}

export async function urlToStudioImage(url: string, name: string): Promise<StudioImage> {
  const blob = await fetch(url).then((r) => r.blob())
  return toStudioImage(new File([blob], name, { type: blob.type }))
}

export const formatSize = (bytes: number) =>
  bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} МБ` : `${Math.round(bytes / 1024)} КБ`
