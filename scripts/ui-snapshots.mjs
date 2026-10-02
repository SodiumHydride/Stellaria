// Screenshots of every main screen against a running dev backend, for visual review.
//
//   node scripts/ui-snapshots.mjs --url http://127.0.0.1:1420 --backend http://127.0.0.1:8765 --out shots
//
// Uses playwright-core from desktop/node_modules and a local Chromium
// (CHROMIUM_PATH, or the path Playwright installs to).
import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join, resolve } from 'node:path'

const require = createRequire(resolve('desktop/package.json'))
const { chromium } = require('playwright-core')

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, cur, i, arr) => (cur.startsWith('--') ? [...acc, [cur.slice(2), arr[i + 1]]] : acc), []),
)
const url = args.url ?? 'http://127.0.0.1:1420'
const backend = args.backend ?? 'http://127.0.0.1:8765'
const out = args.out ?? 'shots'
const lang = args.lang ?? 'zh-CN'
const theme = args.theme ?? 'light'
mkdirSync(out, { recursive: true })

const exe = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
const browser = await chromium.launch({ executablePath: exe, args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
await page.addInitScript(([l, th]) => {
  localStorage.setItem('stellaria.lang', l)
  localStorage.setItem('stellaria.theme', th)
}, [lang, theme])

const base = `${url}/?backend=${encodeURIComponent(backend)}&token=dev`
const api = async (path, init) => {
  const res = await fetch(`${backend}/api${path}`, { ...init, headers: { Authorization: 'Bearer dev', 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  return res.json()
}

async function shot(name, hash, wait = 1200) {
  await page.goto(`${base}#${hash}`)
  await page.waitForTimeout(wait)
  await page.screenshot({ path: join(out, `${name}.png`) })
  console.log('shot', name)
}

const characters = await api('/characters')
const projects = await api('/projects')
const first = characters[0]

await shot('01-voices', '/voices', 2500)
if (first) {
  await shot('02-material', `/voices/${first.id}/material`, 2500)
  await shot('03-training', `/voices/${first.id}/training`, 3000)
  await shot('04-models', `/voices/${first.id}/models`, 1500)
}
await shot('05-create', '/create', 1500)
if (projects[0]) await shot('06-canvas', `/create/${projects[0].id}`, 3000)
await shot('07-resources', '/resources', 1500)
await shot('08-settings', '/settings', 1200)

// command palette and dock
await page.goto(`${base}#/voices`)
await page.waitForTimeout(1500)
await page.keyboard.press('Control+K')
await page.waitForTimeout(500)
await page.screenshot({ path: join(out, '09-palette.png') })
await page.keyboard.press('Escape')
await page.click('.dock').catch(() => {})
await page.waitForTimeout(600)
await page.screenshot({ path: join(out, '10-dock.png') })

console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors')
await browser.close()
