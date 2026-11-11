#!/bin/bash
# Build dist/Subvox.dmg: Release build, ad-hoc signed, drag-to-Applications window with background.
# ponytail: ad-hoc signature (no Developer ID), so friends need right-click > Open once. Upgrade path:
# Developer ID cert + `xcrun notarytool submit --wait` + `xcrun stapler staple`.
set -euo pipefail
cd "$(dirname "$0")/.."
xcodegen generate -q
xcodebuild -project Subvox.xcodeproj -scheme Subvox -configuration Release -derivedDataPath build/rel \
  CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= PROVISIONING_PROFILE_SPECIFIER= build -quiet
APP=build/rel/Build/Products/Release/Subvox.app
codesign --force --deep -s - "$APP"

rm -rf dist && mkdir dist
APP="$APP" uv run -q --with dmgbuild dmgbuild -s dmg/settings.py Subvox dist/Subvox.dmg
echo "built dist/Subvox.dmg"
