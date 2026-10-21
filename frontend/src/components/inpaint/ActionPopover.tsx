import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { ArrowUp, BrickWall, ImagePlus, Sparkles, Trash2, X } from 'lucide-react'
import { editKind, type InpaintMode } from '../../lib/inpaint'

import { useScene, type Scene } from '../../lib/scene'

// со словом-командой: «добавь» — новый объект в натуральную величину, «замени / перекрась» — меняется то, что есть
const SUGGESTIONS: Record<Scene, string[]> = {
  exterior: ['Добавь растения', 'Добавь людей', 'Добавь автомобиль', 'Добавь фонарь', 'Замени на газон', 'Перекрась в белый'],
  interior: ['Добавь растения', 'Добавь диван', 'Добавь картину', 'Добавь ковёр', 'Замени на паркет', 'Перекрась в белый'],
}

// уточнение для удаления: что именно убрать внутри выделения, остальное остаётся
const REMOVE_SUGGESTIONS: Record<Scene, string[]> = {
  exterior: ['Люди', 'Машины', 'Деревья', 'Столбы и провода', 'Знаки', 'Мусор'],
  interior: ['Люди', 'Мебель', 'Декор', 'Растения', 'Провода', 'Светильник'],
}

// материал для выделенной поверхности: тип, цвет, рисунок — поверхность сохраняет форму, швы и проёмы
const MATERIAL_SUGGESTIONS: Record<Scene, string[]> = {
  exterior: ['Дерево — венге', 'Белая штукатурка', 'Клинкерный кирпич', 'Натуральный камень', 'Архитектурный бетон', 'Тёмный металл'],
  interior: ['Дубовый паркет', 'Белая штукатурка', 'Микроцемент', 'Мрамор', 'Керамогранит', 'Деревянные панели'],
}

const EDIT_HINT = {
  none: 'Начните с «добавь» или «измени» — так точнее',
  add: 'Добавится новый объект в натуральную величину',
  change: 'Изменится выделенное: материал и цвет — на месте, другой объект — в своём размере',
}

const MATERIAL_PLACEHOLDER: Record<Scene, string> = {
  exterior: 'Например: вертикальная доска из лиственницы, серая',
  interior: 'Например: светлый дуб ёлочкой, матовый лак',
}

const PLACEHOLDER: Record<Scene, string> = {
  exterior: 'Например: добавь деревянную террасу с шезлонгами',
  interior: 'Например: добавь мягкий диван из бежевого букле',
}

interface Props {
  mode: InpaintMode
  prompt: string
  onPrompt: (v: string) => void
  /** Образец материала (только «Материалы») */
  texture?: File | null
  onTexture?: (f: File | null) => void
  /** Ширина образца в сантиметрах (строка поля; пусто — оценит бэкенд) */
  textureWidth?: string
  onTextureWidth?: (v: string) => void
  onConfirm: () => void
  onReset: () => void
  /** Позиция «носика» относительно сцены */
  x: number
  y: number
  placement: 'top' | 'bottom'
}

export const POPOVER_WIDTH = 340

export function ActionPopover({
  mode, prompt, onPrompt, texture, onTexture, textureWidth, onTextureWidth, onConfirm, onReset, x, y, placement,
}: Props) {
  const input = useRef<HTMLTextAreaElement>(null)
  const { scene } = useScene()
  const suggestions = mode === 'edit' ? SUGGESTIONS[scene] : mode === 'material' ? MATERIAL_SUGGESTIONS[scene] : REMOVE_SUGGESTIONS[scene]
  useEffect(() => {
    if (mode !== 'remove') input.current?.focus({ preventScroll: true })
  }, [mode])

  const canConfirm = mode === 'remove' || prompt.trim().length > 0 || (mode === 'material' && !!texture)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.94, y: placement === 'bottom' ? -8 : 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96, transition: { duration: 0.15 } }}
      transition={{ type: 'spring', stiffness: 420, damping: 32 }}
      style={{
        left: x,
        top: y,
        width: POPOVER_WIDTH,
        translateX: '-50%',
        translateY: placement === 'top' ? '-100%' : '0%',
        transformOrigin: placement === 'top' ? 'bottom center' : 'top center',
      }}
      data-no-zoom
      className="absolute z-30 rounded-[22px] bg-surface/[0.88] p-4 backdrop-blur-2xl backdrop-saturate-150 shadow-float ring-1 ring-fg/[0.06]"
      onPointerDown={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !e.shiftKey && canConfirm) { e.preventDefault(); onConfirm() }
        e.stopPropagation()
      }}
    >
      <div className="flex items-start gap-3">
        <span
          className="grid size-9 shrink-0 place-items-center rounded-full text-white"
          style={{
            background: mode === 'edit' ? 'linear-gradient(135deg,#5a9bff,#6a5cff)'
              : mode === 'material' ? 'linear-gradient(135deg,#fbbf24,#f59e0b)'
              : 'linear-gradient(135deg,#ff6b6b,#ff3b5c)',
          }}
        >
          {mode === 'edit' ? <Sparkles className="size-4" /> : mode === 'material' ? <BrickWall className="size-4" /> : <Trash2 className="size-4" />}
        </span>
        <div className="min-w-0 pt-0.5">
          <div className="text-[15px] font-semibold tracking-[-0.01em]">
            {mode === 'edit' ? 'Что сделать с областью?' : mode === 'material' ? 'Какой материал положить?' : 'Удалить выделенное?'}
          </div>
          <div className="text-[12.5px] leading-snug text-mute">
            {mode === 'edit'
              ? EDIT_HINT[editKind(prompt) ?? 'none']
              : mode === 'material'
                ? texture
                  ? 'Образец ляжет 1 в 1 — по граням, в перспективе и свете кадра'
                  : 'Поверхность сохранит форму, швы и проёмы — сменится только материал'
                : prompt.trim()
                ? 'Удалится только это — остальное в выделении останется'
                : 'Всё выделенное и его тень заменит окружающий фон'}
          </div>
        </div>
      </div>

      {mode === 'remove' && (
        <div className="mt-3.5 text-[12px] font-medium text-faint">Что именно удалить? Необязательно</div>
      )}
      {mode === 'material' && onTexture && (
        <TextureDrop texture={texture ?? null} onTexture={onTexture} width={textureWidth ?? ''} onWidth={onTextureWidth} />
      )}
      <textarea
        ref={input}
        value={prompt}
        onChange={(e) => onPrompt(e.target.value)}
        rows={mode === 'remove' ? 1 : 2}
        maxLength={300}
        placeholder={
          mode === 'edit' ? PLACEHOLDER[scene]
            : mode === 'material' ? (texture ? 'Уточнение, необязательно: например, доски вертикально' : MATERIAL_PLACEHOLDER[scene])
            : 'Например: люди — без уточнения удалится всё'
        }
        className={`${mode === 'remove' ? 'mt-1.5' : mode === 'material' ? 'mt-2' : 'mt-3.5'} w-full resize-none rounded-[14px] bg-fg/[0.04] px-3.5 py-2.5 text-[14px] leading-snug outline-none ring-accent/40 transition-shadow placeholder:text-faint focus:bg-surface focus:ring-2`}
      />
      <div className="mt-2 flex flex-wrap gap-1.5">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => onPrompt(mode === 'remove' && prompt === s ? '' : s)}
            className={`rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${
              prompt === s ? 'bg-ink text-on-ink' : 'bg-fg/[0.05] text-ink-2 hover:bg-fg/[0.09]'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-between gap-2">
        <button onClick={onReset} className="rounded-full px-3.5 py-2 text-[13px] font-medium text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink">
          Сбросить
        </button>
        <motion.button
          whileTap={canConfirm ? { scale: 0.96 } : undefined}
          onClick={onConfirm}
          disabled={!canConfirm}
          className={`flex items-center gap-1.5 rounded-full px-4 py-2 text-[13.5px] font-medium text-white transition-all duration-200 disabled:bg-fg/[0.08] disabled:text-faint ${
            mode === 'edit' ? 'bg-accent hover:bg-accent-hover' : mode === 'material' ? 'bg-[#f59e0b] hover:bg-[#e8900a]' : 'bg-[#ff3b30] hover:bg-[#ff2d20]'
          }`}
        >
          {mode !== 'remove' ? (
            <>Применить <ArrowUp className="size-3.5" strokeWidth={2.5} /></>
          ) : (
            <><Trash2 className="size-3.5" /> Удалить</>
          )}
        </motion.button>
      </div>
    </motion.div>
  )
}

const TEXTURE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const TEXTURE_MAX = 20 * 1024 * 1024

/** Образец материала: перетащить файл или выбрать по клику; превью с кнопкой «убрать». */
function TextureDrop({ texture, onTexture, width, onWidth }: {
  texture: File | null
  onTexture: (f: File | null) => void
  width: string
  onWidth?: (v: string) => void
}) {
  const file = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const preview = useMemo(() => (texture ? URL.createObjectURL(texture) : null), [texture])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  function take(f: File | undefined) {
    if (!f) return
    if (!TEXTURE_TYPES.includes(f.type)) return setError('Нужна картинка: JPG, PNG или WebP')
    if (f.size > TEXTURE_MAX) return setError('Файл больше 20 МБ')
    setError(null)
    onTexture(f)
  }

  return (
    <div className="mt-3.5">
      <div
        role="button"
        tabIndex={0}
        onClick={() => file.current?.click()}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.current?.click() } }}
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); e.stopPropagation(); setOver(false); take(e.dataTransfer.files[0]) }}
        className={`flex cursor-pointer items-center gap-3 rounded-[14px] p-2 transition-colors ${
          over ? 'bg-[#f59e0b]/15 ring-2 ring-[#f59e0b]' : texture ? 'bg-fg/[0.04]' : 'border border-dashed border-fg/20 bg-fg/[0.03] hover:bg-fg/[0.07]'
        }`}
      >
        {preview ? (
          <img src={preview} alt="" className="size-12 shrink-0 rounded-[10px] object-cover ring-1 ring-fg/10" />
        ) : (
          <span className="grid size-12 shrink-0 place-items-center rounded-[10px] bg-fg/[0.05] text-mute">
            <ImagePlus className="size-5" strokeWidth={1.75} />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium text-ink">{texture ? texture.name : 'Образец материала'}</div>
          <div className="text-[12px] leading-snug text-faint">
            {texture ? 'Нажмите, чтобы заменить' : 'Перетащите текстуру сюда или выберите файл'}
          </div>
        </div>
        {texture && (
          <button
            aria-label="Убрать образец"
            onClick={(e) => { e.stopPropagation(); onTexture(null) }}
            className="grid size-7 shrink-0 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.08] hover:text-ink"
          >
            <X className="size-3.5" />
          </button>
        )}
        <input
          ref={file}
          type="file"
          accept={TEXTURE_TYPES.join(',')}
          className="hidden"
          onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }}
        />
      </div>
      {texture && onWidth && (
        <label className="mt-2 flex items-center gap-2 text-[12.5px] text-mute">
          <span className="flex-1">Ширина образца</span>
          <input
            type="number"
            inputMode="decimal"
            min={1}
            max={5000}
            value={width}
            onChange={(e) => onWidth(e.target.value)}
            placeholder="авто"
            className="w-20 rounded-[10px] bg-fg/[0.04] px-2.5 py-1.5 text-right text-[13px] text-ink outline-none ring-accent/40 placeholder:text-faint focus:bg-surface focus:ring-2"
          />
          <span>см</span>
        </label>
      )}
      {error && <div className="mt-1.5 text-[12px] text-[#ff3b30]">{error}</div>}
    </div>
  )
}
