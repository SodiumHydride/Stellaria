import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query'
import { lazy, Suspense, useEffect, useState, type CSSProperties } from 'react'
import { createHashRouter, Navigate, Outlet, RouterProvider, useLocation, useNavigate } from 'react-router-dom'
import { useActiveJobs, usePreferences } from '../api/hooks'
import { startLive } from '../api/live'
import { useConnection } from '../api/client'
import { CommandPalette, ShortcutsDialog } from '../components/CommandPalette'
import { DropOverlay } from '../components/DropOverlay'
import { RouteError } from './RouteError'
import { Backdrop } from '../components/Backdrop'
import { TaskDock } from '../components/TaskDock'
import { spaceOf, TitleBar } from '../components/TitleBar'
import { Toasts } from '../components/Toasts'
import { ConfirmHost, TooltipProvider } from '../components/ui'
import { useLang } from '../i18n'
import { Onboarding } from '../features/onboarding/Onboarding'
import { TermsGate } from '../features/terms/Terms'
import { MissingModelsBar } from '../components/MissingModelsBar'
import { BootScreen } from './BootScreen'
import { useBoot } from './boot'
import { useJobNotifier } from './useJobNotifier'
import { useTrayStatus } from './useTrayStatus'
import { useCloseRequests } from './useCloseRequests'
import { useAutoRelease } from './useAutoRelease'
import './app.css'

const VoicesSpace = lazy(() => import('../features/characters/VoicesSpace'))
const CreateSpace = lazy(() => import('../features/canvas/CreateSpace'))
const ResourcesView = lazy(() => import('../features/resources/ResourcesView'))
const SettingsView = lazy(() => import('../features/settings/SettingsView'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 15_000 },
  },
})

function useEffectsLevel() {
  const { data: prefs } = usePreferences()
  const training = useActiveJobs((j) => j.type === 'train' && j.status === 'running').length > 0
  const lowPower = !!prefs?.lowPowerWhileTraining && training
  useEffect(() => {
    document.documentElement.dataset.glass = prefs && !prefs.glass ? 'off' : 'on'
    document.documentElement.dataset.lowPower = lowPower ? 'on' : 'off'
  }, [prefs, lowPower])
  useEffect(() => {
    if (prefs?.language && prefs.language !== useLang.getState().lang) useLang.getState().setLang(prefs.language)
  }, [prefs?.language])
}

function GlobalShortcuts() {
  const nav = useNavigate()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return
      const target = e.target as HTMLElement
      if (target.closest('input, textarea, [contenteditable="true"]')) return
      if (e.key === '1') nav('/voices')
      else if (e.key === '2') nav('/create')
      else if (e.key === '3') nav('/resources')
      else if (e.key === ',') nav('/settings')
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [nav])
  return null
}

const SPACE_ORDER = { voices: 0, create: 1, resources: 2, settings: 3 } as const

/** The current space; switching to another slides it in from the side it sits on in the title
 *  bar's switcher. Moving within a space (another character, another project) does not. */
function SpaceStage() {
  const space = spaceOf(useLocation().pathname)
  const [stage, setStage] = useState({ space, dir: 1 })
  if (stage.space !== space) setStage({ space, dir: SPACE_ORDER[space] >= SPACE_ORDER[stage.space] ? 1 : -1 })
  return (
    <div key={space} className="space-stage" style={{ '--enter-dir': stage.space === space ? stage.dir : 1 } as CSSProperties}>
      <Outlet />
    </div>
  )
}

function Layout() {
  const qc = useQueryClient()
  const url = useConnection((s) => s.url)
  useEffect(() => (url ? startLive(qc) : undefined), [qc, url])
  useJobNotifier()
  useTrayStatus()
  useCloseRequests()
  useAutoRelease()
  useEffectsLevel()

  return (
    <>
      <Backdrop />
      <div className="app">
        <TitleBar />
        <MissingModelsBar />
        <main className="app-main">
          <Suspense fallback={null}>
            <SpaceStage />
          </Suspense>
        </main>
      </div>
      <TaskDock />
      <Toasts />
      <CommandPalette />
      <ShortcutsDialog />
      <DropOverlay />
      <ConfirmHost />
      <GlobalShortcuts />
      <TermsGate />
    </>
  )
}

const router = createHashRouter([
  {
    path: '/',
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      {
        // a crash inside a screen keeps the window chrome around it
        errorElement: <RouteError />,
        children: [
          { index: true, element: <Navigate to="/voices" replace /> },
          { path: 'voices', element: <VoicesSpace /> },
          { path: 'voices/:characterId', element: <VoicesSpace /> },
          { path: 'voices/:characterId/:tab', element: <VoicesSpace /> },
          { path: 'create', element: <CreateSpace /> },
          { path: 'create/:projectId', element: <CreateSpace /> },
          { path: 'resources', element: <ResourcesView /> },
          { path: 'settings', element: <SettingsView /> },
          { path: '*', element: <Navigate to="/voices" replace /> },
        ],
      },
    ],
  },
])

function Root() {
  const phase = useBoot((s) => s.phase)
  const start = useBoot((s) => s.start)
  useEffect(() => {
    void start()
  }, [start])

  if (phase === 'ready') return <RouterProvider router={router} />
  if (phase === 'onboarding') return <Onboarding />
  return <BootScreen />
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={350}>
        <Root />
      </TooltipProvider>
    </QueryClientProvider>
  )
}
