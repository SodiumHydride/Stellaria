import { Plus, Upload } from 'lucide-react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'
import { useCharacters } from '../../api/hooks'
import { Empty } from '../../components/ui'
import { useT } from '../../i18n'
import { CharacterSidebar, createCharacter } from './CharacterSidebar'
import { CharacterWorkspace } from './CharacterWorkspace'
import './voices.css'

/** 角色: the list stays on the left, so switching voices never leaves the page. */
export default function VoicesSpace() {
  const { characterId, tab } = useParams()
  return (
    <div className="voices-space">
      <CharacterSidebar selectedId={characterId} />
      <section className="voices-main">{characterId ? <CharacterWorkspace key={characterId} characterId={characterId} tab={tab} /> : <Landing />}</section>
    </div>
  )
}

/** No character in the URL: reopen the last one, or invite the first. */
function Landing() {
  const t = useT()
  const nav = useNavigate()
  const { data: characters, isLoading } = useCharacters(false)
  if (isLoading || !characters) return null
  if (characters.length > 0) {
    const last = [...characters].sort((a, b) => (b.lastOpenedAt ?? b.updatedAt).localeCompare(a.lastOpenedAt ?? a.updatedAt))[0]
    return <Navigate to={`/voices/${last.id}`} replace />
  }
  const onNew = async () => {
    const c = await createCharacter(t.lang === 'zh-CN' ? '新角色' : 'New voice', [])
    if (c) nav(`/voices/${c.id}/material?rename=1`)
  }
  return (
    <div className="voices-landing glass">
      <Empty icon={<Upload />} title={t('voices.empty.title')} body={t('voices.empty.body')}>
        <button className="btn btn-primary" onClick={onNew}>
          <Plus />
          {t('voices.new')}
        </button>
      </Empty>
    </div>
  )
}
