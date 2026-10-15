/** Квадратная кнопка панели инструментов с всплывающей подсказкой и сочетанием клавиш */
export function ToolButton({
  children, label, shortcut, onClick, active, disabled, side = 'right',
}: {
  children: React.ReactNode; label: string; shortcut: string; onClick: () => void; active?: boolean; disabled?: boolean
  /** с какой стороны кнопки показывать подсказку */
  side?: 'left' | 'right'
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={`group relative isolate grid size-10 place-items-center rounded-xl transition-colors duration-200 disabled:opacity-35 ${
        active ? 'text-on-ink' : 'text-ink-2 hover:bg-fg/[0.06] hover:text-ink'
      }`}
    >
      {children}
      <span className={`pointer-events-none absolute top-1/2 z-20 flex -translate-y-1/2 ${
        side === 'right' ? 'left-full ml-3 -translate-x-1' : 'right-full mr-3 translate-x-1'
      } items-center gap-2 whitespace-nowrap rounded-lg bg-ink px-2.5 py-1.5 text-[12px] font-medium text-on-ink opacity-0 shadow-lg transition-all duration-200 ease-apple group-hover:translate-x-0 group-hover:opacity-100 group-disabled:hidden`}>
        {label}
        <span className="text-on-ink/50">{shortcut}</span>
      </span>
    </button>
  )
}
