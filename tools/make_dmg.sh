#!/bin/bash
# Build dist/Subvox.dmg: Release build, drag-to-Applications window with background.
# Default: ad-hoc signature (no Developer ID), so friends need right-click > Open once.
# Notarized: set DEVELOPER_ID (a signing identity, e.g. "Developer ID Application: Name (TEAMID)") and
# NOTARY_PROFILE (saved once with `xcrun notarytool store-credentials`). The app is then signed with the
# hardened runtime and Subvox/Subvox.entitlements, and the DMG is signed, notarized and stapled.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVELOPER_ID="${DEVELOPER_ID:-}"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"
fail() { echo "make_dmg: $*" >&2; exit 1; }
# Check the credentials before the build, so a typo fails in seconds, not after a full build.
if [ -n "$DEVELOPER_ID$NOTARY_PROFILE" ]; then
  [ -n "$DEVELOPER_ID" ] && [ -n "$NOTARY_PROFILE" ] ||
    fail "set both DEVELOPER_ID and NOTARY_PROFILE for a notarized DMG, or neither for an ad-hoc one"
  identities=$(security find-identity -v -p codesigning)
  grep -qF -- "$DEVELOPER_ID" <<<"$identities" ||
    fail "no valid signing identity matching DEVELOPER_ID=\"$DEVELOPER_ID\" in the keychain"
  xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null ||
    fail "notary profile \"$NOTARY_PROFILE\" failed (see above). Save it with:" \
      "xcrun notarytool store-credentials \"$NOTARY_PROFILE\" --apple-id ... --team-id ... --password ..."
fi

xcodegen generate -q
xcodebuild -project Subvox.xcodeproj -scheme Subvox -configuration Release -derivedDataPath build/rel \
  CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= PROVISIONING_PROFILE_SPECIFIER= build -quiet
APP=build/rel/Build/Products/Release/Subvox.app
if [ -n "$DEVELOPER_ID" ]; then
  # Hardened runtime blocks the mic unless the audio-input entitlement is present. Motion needs none.
  codesign --force --options runtime --timestamp --entitlements Subvox/Subvox.entitlements \
    -s "$DEVELOPER_ID" "$APP"
else
  codesign --force --deep -s - "$APP"
fi

rm -rf dist && mkdir dist
APP="$APP" uv run -q --with dmgbuild dmgbuild -s dmg/settings.py Subvox dist/Subvox.dmg
if [ -z "$DEVELOPER_ID" ]; then
  echo "built dist/Subvox.dmg"
  exit 0
fi

codesign --force --timestamp -s "$DEVELOPER_ID" dist/Subvox.dmg
echo "notarizing dist/Subvox.dmg (usually a few minutes, gives up after 1 hour)"
# Trust the reported status, not the exit code, and print Apple's log if it isn't Accepted.
result=$(xcrun notarytool submit dist/Subvox.dmg --keychain-profile "$NOTARY_PROFILE" \
  --wait --timeout 1h --output-format plist) || true
status=$(plutil -extract status raw -o - - <<<"$result" 2>/dev/null) || status=""
if [ "$status" != Accepted ]; then
  id=$(plutil -extract id raw -o - - <<<"$result" 2>/dev/null) || id=""
  [ -z "$id" ] || xcrun notarytool log "$id" --keychain-profile "$NOTARY_PROFILE" >&2 || true
  fail "notarization failed (status: ${status:-none}). See the messages above."
fi
xcrun stapler staple dist/Subvox.dmg
echo "built dist/Subvox.dmg (Developer ID signed, notarized, stapled)"
