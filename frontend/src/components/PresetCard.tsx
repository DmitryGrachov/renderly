import { motion } from 'motion/react'
import { Check } from 'lucide-react'
import type { Preset } from '../lib/presets'

interface Props {
  preset: Preset
  selected: boolean
  onSelect: () => void
  disabled?: boolean
  wide?: boolean
}

export function PresetCard({ preset, selected, onSelect, disabled, wide }: Props) {
  return (
    <motion.button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      whileTap={{ scale: 0.97 }}
      className={`group relative overflow-hidden rounded-[18px] bg-white text-left transition-shadow duration-300 ease-apple disabled:opacity-50 ${
        selected
          ? 'shadow-[0_0_0_2px_var(--color-ink),0_8px_24px_-8px_rgba(0,0,0,0.25)]'
          : 'shadow-[0_0_0_1px_rgba(0,0,0,0.06)] hover:shadow-[0_0_0_1px_rgba(0,0,0,0.1),0_8px_24px_-12px_rgba(0,0,0,0.2)]'
      } ${wide ? 'col-span-2 flex items-stretch' : ''}`}
    >
      <div className={`relative overflow-hidden ${wide ? 'w-[92px] shrink-0' : 'h-[76px]'}`} style={{ background: preset.sky }}>
        {preset.sun && (
          <span
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full transition-transform duration-700 ease-apple group-hover:scale-110"
            style={{
              left: preset.sun.x,
              top: preset.sun.y,
              width: preset.sun.size,
              height: preset.sun.size,
              background: preset.sun.color,
              boxShadow: `0 0 ${preset.sun.size}px ${preset.sun.size / 3}px ${preset.sun.color}`,
            }}
          />
        )}
        {preset.room && <Room preset={preset} />}
        {/* силуэт здания */}
        {!wide && !preset.room && (
          <svg viewBox="0 0 200 40" preserveAspectRatio="none" className="absolute inset-x-0 bottom-0 h-[38%] w-full">
            <path d="M0 40 V26 H34 V14 H70 V22 H96 V6 H134 V18 H160 V24 H200 V40Z" fill="rgba(0,0,0,0.16)" />
          </svg>
        )}
        {wide && <span className="absolute inset-0 backdrop-blur-[6px]" />}
      </div>

      <div className={`p-3 ${wide ? 'flex-1 py-3.5 pl-4' : ''}`}>
        <div className="text-[13.5px] font-semibold leading-tight tracking-[-0.01em]">{preset.title}</div>
        <div className="mt-1 text-[11px] font-medium tracking-wide text-faint">{preset.meta}</div>
        {wide && <div className="mt-1.5 text-[12px] leading-snug text-mute">{preset.caption}</div>}
      </div>

      <motion.span
        initial={false}
        animate={{ scale: selected ? 1 : 0, opacity: selected ? 1 : 0 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        className="absolute right-2 top-2 grid size-5 place-items-center rounded-full bg-ink text-white shadow-md"
      >
        <Check className="size-3" strokeWidth={3} />
      </motion.span>
    </motion.button>
  )
}

/** Превью интерьера: стена, окно с «небом» пресета, пятно света на полу */
function Room({ preset }: { preset: Preset }) {
  const room = preset.room!
  return (
    <div className="absolute inset-0" style={{ background: room.wall }}>
      {/* пол */}
      <div className="absolute inset-x-0 bottom-0 h-[26%] bg-black/[0.07]" />
      {/* окно */}
      <div
        className="absolute left-[30%] top-[14%] h-[52%] w-[40%] overflow-hidden rounded-[3px] ring-[3px] ring-white/70 transition-transform duration-700 ease-apple group-hover:scale-[1.04]"
        style={{ background: preset.sky }}
      >
        <div className="absolute inset-y-0 left-1/2 w-[3px] -translate-x-1/2 bg-white/70" />
      </div>
      {/* пятно света из окна */}
      <div
        className="absolute bottom-[2%] left-[22%] h-[24%] w-[56%] blur-[2px]"
        style={{ background: room.patch, clipPath: 'polygon(16% 0, 84% 0, 100% 100%, 0 100%)' }}
      />
      {/* лампа */}
      {room.lamp && (
        <span
          className="absolute right-[12%] top-[34%] size-2.5 rounded-full"
          style={{ background: room.lamp, boxShadow: `0 0 18px 8px ${room.lamp}99` }}
        />
      )}
    </div>
  )
}
