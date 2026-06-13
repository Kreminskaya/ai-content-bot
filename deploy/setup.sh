#!/bin/bash
# =================================================================
#  setup.sh — install AI Content Bot on Ubuntu 24.04
#  Run as root:
#    bash /opt/ai-telegram-assistant/deploy/setup.sh
# =================================================================

# Strip Windows line endings if the script was transferred from Windows
sed -i 's/\r//' "$0"

set -e  # stop on any error

APP_DIR="/opt/ai-telegram-assistant"
SERVICE_NAME="ai-telegram-bot"
SERVICE_SRC="$APP_DIR/deploy/ai-telegram-bot.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "┌──────────────────────────────────────────┐"
echo "│   AI Content Bot — server installation   │"
echo "└──────────────────────────────────────────┘"
echo ""

# 1. System packages
echo "[1/6] Updating system and installing Python..."
apt-get update -q
apt-get install -y python3 python3-pip python3-venv

# 2. Virtual environment (recreate if broken)
echo "[2/6] Creating virtualenv..."
cd "$APP_DIR"
python3 -m venv venv

# 3. Python dependencies
echo "[3/6] Installing Python dependencies (2-3 minutes)..."
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# 4. Media folder
echo "[4/6] Creating media/ directory..."
mkdir -p "$APP_DIR/media"

# 5. Systemd service
echo "[5/6] Installing systemd service..."
# Strip Windows line endings from the service file
sed -i 's/\r//' "$SERVICE_SRC"
cp "$SERVICE_SRC" "$SERVICE_DST"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# 6. Done
echo ""
echo "✅ [6/6] Installation complete!"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Next steps:"
echo ""
echo "  1. Authorise the Telethon userbot (one time only):"
echo "       cd $APP_DIR && venv/bin/python auth_userbot.py"
echo "     → enter the code from Telegram → wait for 'Session saved'"
echo ""
echo "  2. Start the bot:"
echo "       systemctl start $SERVICE_NAME"
echo ""
echo "  3. Check the logs:"
echo "       journalctl -u $SERVICE_NAME -f"
echo "═══════════════════════════════════════════════════════"
