#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

APP_NAME="${APP_NAME:-CeDiTT1.0}"
APP_VERSION="${APP_VERSION:-3.0.0}"
APP_BUNDLE_ID="${APP_BUNDLE_ID:-com.vincenzobarone.ceditt}"
ICON_PATH="${ICON_PATH:-$BASE_DIR/assets/icons/app_icon.icns}"
PYI_CONFIG_DIR="${PYI_CONFIG_DIR:-$BASE_DIR/.pyinstaller}"

PY_BIN="${PY_BIN:-$(command -v python3)}"
if [[ -z "$PY_BIN" || ! -x "$PY_BIN" ]]; then
  echo "python3 not found." >&2
  exit 1
fi

if ! "$PY_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller is not installed in $PY_BIN." >&2
  echo "Install it with: $PY_BIN -m pip install pyinstaller" >&2
  exit 1
fi

mkdir -p "$PYI_CONFIG_DIR"
rm -rf "$BASE_DIR/build" "$BASE_DIR/dist" "$BASE_DIR/${APP_NAME}.spec"

PYINSTALLER_CONFIG_DIR="$PYI_CONFIG_DIR" \
"$PY_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --icon "$ICON_PATH" \
  --osx-bundle-identifier "$APP_BUNDLE_ID" \
  --contents-directory "_internal" \
  ceditt_gui.py

APP_PATH="$BASE_DIR/dist/${APP_NAME}.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Build failed: ${APP_PATH} not found." >&2
  exit 1
fi

if /usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1; then
  /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $APP_VERSION" "$APP_PATH/Contents/Info.plist" >/dev/null
else
  /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $APP_VERSION" "$APP_PATH/Contents/Info.plist" >/dev/null
fi

if /usr/libexec/PlistBuddy -c "Print :CFBundleVersion" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1; then
  /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $APP_VERSION" "$APP_PATH/Contents/Info.plist" >/dev/null
else
  /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $APP_VERSION" "$APP_PATH/Contents/Info.plist" >/dev/null
fi

if /usr/libexec/PlistBuddy -c "Print :LSMinimumSystemVersion" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1; then
  /usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion 11.0" "$APP_PATH/Contents/Info.plist" >/dev/null
else
  /usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string 11.0" "$APP_PATH/Contents/Info.plist" >/dev/null
fi

echo "Built standalone app: $APP_PATH"
