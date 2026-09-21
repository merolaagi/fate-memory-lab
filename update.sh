#!/bin/bash
set -e
DEST="$(cd "$(dirname "$0")" && pwd)"
ZIP="$1"
if [ -z "$ZIP" ]; then
  ZIP="$(ls -t "$HOME"/Downloads/fate-memory-lab-v*.zip 2>/dev/null | head -1)"
fi
[ -f "$ZIP" ] || { echo "No fate-memory-lab-v*.zip found in ~/Downloads. Pass the zip path: ./update.sh /path/to/file.zip"; exit 1; }
TMP="$(mktemp -d)"
unzip -q "$ZIP" -d "$TMP"
SRC="$TMP/fate-memory-lab"
[ -d "$SRC/app" ] || { echo "$ZIP is not a Fate Memory Lab package"; rm -rf "$TMP"; exit 1; }
OLD="$(cat "$DEST/VERSION" 2>/dev/null || echo none)"
rsync -a --exclude data --exclude .venv --exclude logs --exclude .git "$SRC/" "$DEST/"
rm -rf "$TMP"
cd "$DEST"
bash install.sh
./service.sh install
echo "Updated $OLD -> $(cat VERSION) from $(basename "$ZIP")"
