#!/bin/bash
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="com.manish.fatememorylab"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
PORT="${FML_PORT:-47431}"
HOST="${FML_HOST:-127.0.0.1}"
case "$1" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents" logs
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$DIR/run.sh</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>EnvironmentVariables</key><dict>
    <key>FML_PORT</key><string>$PORT</string>
    <key>FML_HOST</key><string>$HOST</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DIR/logs/server.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/server.log</string>
</dict></plist>
PL
    launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Service running at http://$HOST:$PORT"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Service removed"
    ;;
  restart)
    launchctl kickstart -k "$DOMAIN/$LABEL"
    echo "Service restarted"
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|pid" || echo "Service not installed"
    ;;
  logs)
    tail -n 80 -f logs/server.log
    ;;
  *)
    echo "Usage: ./service.sh install|uninstall|restart|status|logs"
    ;;
esac
