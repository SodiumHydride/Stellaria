export function Group({ icon, title, children, aside }: { icon: React.ReactNode; title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="set-group">
      <div className="set-group-head">
        {icon}
        <span className="section-title">{title}</span>
        {aside && <span className="spacer" />}
        {aside}
      </div>
      <div className="set-card glass">{children}</div>
    </section>
  )
}

export function Row({ label, hint, children }: { label: React.ReactNode; hint?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="set-row">
      <div className="col" style={{ gap: 2, flex: 1, minWidth: 0 }}>
        <span>{label}</span>
        {hint && <span className="muted set-hint">{hint}</span>}
      </div>
      {children}
    </div>
  )
}
