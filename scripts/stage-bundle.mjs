#!/usr/bin/env node
// Stages everything the Tauri bundle ships next to the shell binary into
// desktop/src-tauri/resources/ (git-ignored):
//
//   engine/          the DDSP-SVC sources in engine/ (no weights, no experiments)
//   backend/         the stellaria_backend package + requirements-runtime.txt
//   manifest.json    resources/manifest.json
//   uv/uv[.exe]      the uv binary for the target triple (pinned, sha256-verified)
//   stage.json       what was staged, for diagnostics
//
// Usage: node scripts/stage-bundle.mjs [--target <rust-triple>] [--uv-version 0.8.17]
//                                                 [--uv-archive <file>] [--skip-uv]
// Env:   STELLARIA_UV_BASE_URL  download base instead of https://github.com/astral-sh/uv/releases/download
//        (e.g. a GitHub proxy); the archive is still verified against the official .sha256.
//
// Node >= 18, no dependencies.

import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import zlib from 'node:zlib'

const UV_VERSION = '0.8.17'
const GITHUB_RELEASES = 'https://github.com/astral-sh/uv/releases/download'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const stellariaDir = path.resolve(scriptDir, '..')
const repoRoot = stellariaDir
const engineDir = path.join(repoRoot, 'engine')
const srcTauri = path.join(stellariaDir, 'desktop', 'src-tauri')
const outDir = path.join(srcTauri, 'resources')
const cacheDir = path.join(srcTauri, 'target', 'stage-cache')

const ENGINE_DIRS = ['ddsp', 'reflow', 'encoder', 'nsf_hifigan', 'optimizer', 'logger', 'configs']
const ENGINE_FILES = ['LICENSE']
// Never shipped: caches, weights, experiments.
const EXCLUDED_NAMES = new Set(['__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.DS_Store', 'tests'])
const EXCLUDED_EXT = new Set(['.pyc', '.pyo', '.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.bin'])

const HOST_TRIPLES = {
  'win32-x64': 'x86_64-pc-windows-msvc',
  'win32-arm64': 'aarch64-pc-windows-msvc',
  'darwin-arm64': 'aarch64-apple-darwin',
  'darwin-x64': 'x86_64-apple-darwin',
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'linux-arm64': 'aarch64-unknown-linux-gnu',
}

function fail(message) {
  console.error(`\n✖ stage-bundle: ${message}`)
  process.exit(1)
}

function parseArgs(argv) {
  const opts = { target: null, uvVersion: UV_VERSION, uvArchive: null, skipUv: false }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    const value = () => {
      const v = argv[++i]
      if (!v) fail(`${arg} needs a value`)
      return v
    }
    if (arg === '--target') opts.target = value()
    else if (arg === '--uv-version') opts.uvVersion = value()
    else if (arg === '--uv-archive') opts.uvArchive = path.resolve(value())
    else if (arg === '--skip-uv') opts.skipUv = true
    else if (arg === '-h' || arg === '--help') {
      console.log('node scripts/stage-bundle.mjs [--target <triple>] [--uv-version X] [--uv-archive file] [--skip-uv]')
      process.exit(0)
    } else fail(`unknown argument ${arg}`)
  }
  opts.target ??= HOST_TRIPLES[`${process.platform}-${process.arch}`]
  if (!opts.target) fail(`no default target for ${process.platform}-${process.arch}; pass --target`)
  return opts
}

const keep = (src) => {
  const base = path.basename(src)
  return !EXCLUDED_NAMES.has(base) && !EXCLUDED_EXT.has(path.extname(base).toLowerCase())
}

function copyTree(src, dest) {
  if (!fs.existsSync(src)) fail(`missing ${path.relative(repoRoot, src)}`)
  fs.cpSync(src, dest, { recursive: true, filter: keep })
}

function countFiles(dir) {
  let n = 0
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    n += entry.isDirectory() ? countFiles(path.join(dir, entry.name)) : 1
  }
  return n
}

function stageEngine() {
  const dest = path.join(outDir, 'engine')
  fs.mkdirSync(dest, { recursive: true })
  const scripts = fs.readdirSync(engineDir).filter((f) => f.endsWith('.py') && fs.statSync(path.join(engineDir, f)).isFile())
  if (!scripts.includes('train_reflow.py')) fail('engine/ has no train_reflow.py (the DDSP-SVC sources are missing)')
  for (const file of [...scripts, ...ENGINE_FILES]) fs.copyFileSync(path.join(engineDir, file), path.join(dest, file))
  for (const dir of ENGINE_DIRS) copyTree(path.join(engineDir, dir), path.join(dest, dir))
  return countFiles(dest)
}

function stageBackend() {
  const src = path.join(stellariaDir, 'backend')
  const pkg = path.join(src, 'stellaria_backend')
  if (!fs.existsSync(path.join(pkg, '__main__.py'))) fail('backend/stellaria_backend/__main__.py is missing')
  if (!fs.existsSync(path.join(src, 'requirements-runtime.txt'))) fail('backend/requirements-runtime.txt is missing')
  const dest = path.join(outDir, 'backend')
  copyTree(pkg, path.join(dest, 'stellaria_backend'))
  // The bootstrapper installs exactly this file (manifest runtime.requirements).
  fs.copyFileSync(path.join(src, 'requirements-runtime.txt'), path.join(dest, 'requirements-runtime.txt'))
  return countFiles(dest)
}

function stageManifest() {
  const src = path.join(stellariaDir, 'resources', 'manifest.json')
  const manifest = JSON.parse(fs.readFileSync(src, 'utf8'))
  if (!manifest?.runtime?.id) fail('resources/manifest.json has no runtime.id')
  fs.copyFileSync(src, path.join(outDir, 'manifest.json'))
  return manifest.runtime.id
}

async function download(url, attempts = 3) {
  let lastError
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const response = await fetch(url, { redirect: 'follow' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return Buffer.from(await response.arrayBuffer())
    } catch (err) {
      lastError = err
      console.warn(`  download attempt ${attempt} failed: ${url}: ${err.cause?.code ?? err.message}`)
      await new Promise((r) => setTimeout(r, 2000 * attempt))
    }
  }
  const proxyHint = process.env.HTTPS_PROXY || process.env.https_proxy
    ? ' (behind a proxy? try NODE_USE_ENV_PROXY=1)' : ''
  fail(`cannot download ${url}: ${lastError?.message}${proxyHint}`)
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')

/** Returns the bytes of the first regular file in a .tar.gz whose basename is `name`. */
function extractFromTarGz(archive, name) {
  const tar = zlib.gunzipSync(archive)
  const text = (buf, start, len) => buf.subarray(start, start + len).toString('utf8').replace(/\0.*$/s, '')
  let offset = 0
  let longName = null
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512)
    if (header.every((b) => b === 0)) break
    let entryName = text(header, 0, 100)
    const size = parseInt(text(header, 124, 12).trim() || '0', 8)
    const type = String.fromCharCode(header[156] || 48)
    if (text(header, 257, 5) === 'ustar') {
      const prefix = text(header, 345, 155)
      if (prefix) entryName = `${prefix}/${entryName}`
    }
    const dataStart = offset + 512
    const data = tar.subarray(dataStart, dataStart + size)
    offset = dataStart + Math.ceil(size / 512) * 512
    if (type === 'L') { longName = data.toString('utf8').replace(/\0.*$/s, ''); continue }
    if (type === 'x') {
      const match = /\d+ path=([^\n]*)\n/.exec(data.toString('utf8'))
      if (match) longName = match[1]
      continue
    }
    if (longName) { entryName = longName; longName = null }
    if ((type === '0' || type === '\0') && path.posix.basename(entryName) === name) return Buffer.from(data)
  }
  fail(`${name} not found in the uv archive`)
}

/** Returns the bytes of the first file in a .zip whose basename is `name`. */
function extractFromZip(archive, name) {
  let eocd = -1
  for (let i = archive.length - 22; i >= Math.max(0, archive.length - 65557); i--) {
    if (archive.readUInt32LE(i) === 0x06054b50) { eocd = i; break }
  }
  if (eocd < 0) fail('the uv archive is not a valid zip')
  const entries = archive.readUInt16LE(eocd + 10)
  let p = archive.readUInt32LE(eocd + 16)
  for (let i = 0; i < entries; i++) {
    if (archive.readUInt32LE(p) !== 0x02014b50) fail('corrupt zip central directory')
    const method = archive.readUInt16LE(p + 10)
    const compSize = archive.readUInt32LE(p + 20)
    const nameLen = archive.readUInt16LE(p + 28)
    const extraLen = archive.readUInt16LE(p + 30)
    const commentLen = archive.readUInt16LE(p + 32)
    const localOffset = archive.readUInt32LE(p + 42)
    const entryName = archive.subarray(p + 46, p + 46 + nameLen).toString('utf8')
    p += 46 + nameLen + extraLen + commentLen
    if (path.posix.basename(entryName) !== name) continue
    if (archive.readUInt32LE(localOffset) !== 0x04034b50) fail('corrupt zip local header')
    const start = localOffset + 30 + archive.readUInt16LE(localOffset + 26) + archive.readUInt16LE(localOffset + 28)
    const data = archive.subarray(start, start + compSize)
    if (method === 0) return Buffer.from(data)
    if (method === 8) return zlib.inflateRawSync(data)
    fail(`unsupported zip compression method ${method}`)
  }
  fail(`${name} not found in the uv archive`)
}

async function stageUv(target, version, archivePath) {
  const windows = target.includes('windows')
  const archiveName = `uv-${target}.${windows ? 'zip' : 'tar.gz'}`
  const base = (process.env.STELLARIA_UV_BASE_URL || GITHUB_RELEASES).replace(/\/+$/, '')
  const official = `${GITHUB_RELEASES}/${version}/${archiveName}`

  console.log(`• uv ${version} for ${target}`)
  const expected = (await download(`${official}.sha256`)).toString('utf8').trim().split(/\s+/)[0].toLowerCase()
  if (!/^[0-9a-f]{64}$/.test(expected)) fail(`unexpected .sha256 content for ${archiveName}`)

  fs.mkdirSync(cacheDir, { recursive: true })
  const cached = archivePath ?? path.join(cacheDir, `${version}-${archiveName}`)
  let archive = fs.existsSync(cached) ? fs.readFileSync(cached) : null
  if (archive && sha256(archive) !== expected) {
    if (archivePath) fail(`${archivePath} does not match the official sha256 ${expected}`)
    archive = null
  }
  if (!archive) {
    archive = await download(`${base}/${version}/${archiveName}`)
    const actual = sha256(archive)
    if (actual !== expected) fail(`sha256 mismatch for ${archiveName}: expected ${expected}, got ${actual}`)
    fs.writeFileSync(cached, archive)
  }

  const exe = windows ? 'uv.exe' : 'uv'
  const binary = windows ? extractFromZip(archive, exe) : extractFromTarGz(archive, exe)
  const dest = path.join(outDir, 'uv')
  fs.mkdirSync(dest, { recursive: true })
  fs.writeFileSync(path.join(dest, exe), binary, { mode: 0o755 })
  return { archive: archiveName, sha256: expected, size: binary.length }
}

function gitCommit() {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch {
    return null
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  console.log(`Staging bundle resources for ${opts.target} → ${path.relative(repoRoot, outDir)}`)
  fs.rmSync(outDir, { recursive: true, force: true })
  fs.mkdirSync(outDir, { recursive: true })

  const engineFiles = stageEngine()
  console.log(`• engine: ${engineFiles} files`)
  const backendFiles = stageBackend()
  console.log(`• backend: ${backendFiles} files`)
  const runtimeId = stageManifest()
  console.log(`• manifest: runtime ${runtimeId}`)
  const uv = opts.skipUv ? null : await stageUv(opts.target, opts.uvVersion, opts.uvArchive)
  if (uv) console.log(`• uv: ${(uv.size / 1e6).toFixed(1)} MB, sha256 ${uv.sha256.slice(0, 12)}… verified`)

  const info = {
    target: opts.target,
    uvVersion: uv ? opts.uvVersion : null,
    uv,
    runtimeId,
    engineFiles,
    backendFiles,
    commit: gitCommit(),
    stagedAt: new Date().toISOString(),
  }
  fs.writeFileSync(path.join(outDir, 'stage.json'), JSON.stringify(info, null, 2) + '\n')
  console.log('✔ staged')
}

main().catch((err) => fail(err?.stack ?? String(err)))
