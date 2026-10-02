import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@xyflow/react/dist/style.css'
import 'uplot/dist/uPlot.min.css'
import './styles/tokens.css'
import './styles/base.css'
import './styles/components.css'
import { App } from './app/App'
import { initTheme } from './state/theme'

initTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
