import clsx from 'clsx'
import type { Character } from '../api/types'
import { mediaUrl } from '../api/client'
import './avatar.css'

/** A character's face: their picture, or their initial on their own muted color. */
export function Avatar({
  character,
  size = 36,
  ring,
  className,
}: {
  character: Pick<Character, 'id' | 'name' | 'color' | 'hasAvatar' | 'avatarVersion' | 'stage'>
  size?: number
  ring?: boolean
  className?: string
}) {
  const initial = Array.from(character.name.trim())[0] ?? '·'
  return (
    <span
      className={clsx('avatar', `stage-${character.stage}`, ring && 'with-ring', className)}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.42), ['--voice' as string]: character.color }}
      aria-hidden
    >
      {character.hasAvatar ? (
        <img src={mediaUrl(`/characters/${character.id}/avatar`, { v: character.avatarVersion })} alt="" draggable={false} />
      ) : (
        <span className="avatar-initial">{initial}</span>
      )}
    </span>
  )
}
