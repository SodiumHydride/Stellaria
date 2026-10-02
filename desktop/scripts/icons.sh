#!/usr/bin/env bash
# Regenerates every app icon from the SVG sources in src-tauri/icons:
#   icon.svg        → Windows .ico, Linux/PNG sizes, Windows Store logos, public/icon.png (favicon)
#   icon-macos.svg  → icon.icns (the same tile on Apple's 824 px grid)
#   tray.svg        → tray-template.png (monochrome menu-bar / tray image)
# Run from stellaria/desktop: bash scripts/icons.sh
set -euo pipefail
cd "$(dirname "$0")/.."
icons=src-tauri/icons
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

npx tauri icon "$icons/icon.svg" -o "$tmp/main" >/dev/null 2>&1
for f in "$tmp"/main/*.png "$tmp"/main/icon.ico; do cp "$f" "$icons/"; done

npx tauri icon "$icons/icon-macos.svg" -o "$tmp/mac" >/dev/null 2>&1
cp "$tmp/mac/icon.icns" "$icons/icon.icns"

npx tauri icon "$icons/tray.svg" -o "$tmp/tray" --png 64 >/dev/null 2>&1
cp "$tmp/tray/64x64.png" "$icons/tray-template.png"

cp "$icons/64x64.png" public/icon.png
echo "icons regenerated"
