#!/bin/bash
# =================================================================
#  setup.sh — установка AI Telegram Bot на Ubuntu 24.04
#  Запускать от root:
#    bash /opt/ai-telegram-assistant/deploy/setup.sh
# =================================================================

# Убрать Windows-переносы строк из самого скрипта (на случай scp с Windows)
sed -i 's/\r//' "$0"

set -e  # остановиться при любой ошибке

APP_DIR="/opt/ai-telegram-assistant"
SERVICE_NAME="ai-telegram-bot"
SERVICE_SRC="$APP_DIR/deploy/ai-telegram-bot.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "┌──────────────────────────────────────────┐"
echo "│   AI Telegram Bot — установка на сервер  │"
echo "└──────────────────────────────────────────┘"
echo ""

# 1. Системные пакеты
echo "[1/6] Обновление системы и установка Python..."
apt-get update -q
apt-get install -y python3 python3-pip python3-venv

# 2. Виртуальное окружение (пересоздаём если повреждено)
echo "[2/6] Создание virtualenv..."
cd "$APP_DIR"
python3 -m venv venv

# 3. Зависимости (без --quiet чтобы видеть ошибки)
echo "[3/6] Установка зависимостей Python (2-3 минуты)..."
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# 4. Папка для медиафайлов
echo "[4/6] Создание папки media/..."
mkdir -p "$APP_DIR/media"

# 5. Systemd-сервис
echo "[5/6] Установка systemd-сервиса..."
# Убрать Windows-переносы строк из service-файла
sed -i 's/\r//' "$SERVICE_SRC"
# Скопировать в системную папку
cp "$SERVICE_SRC" "$SERVICE_DST"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# 6. Итог
echo ""
echo "✅ [6/6] Установка завершена!"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Теперь выполни по порядку:"
echo ""
echo "  1. Авторизуй Telethon (один раз):"
echo "       cd $APP_DIR && venv/bin/python auth_userbot.py"
echo "     → введи код из Telegram → дождись 'Session saved'"
echo ""
echo "  2. Запусти бота:"
echo "       systemctl start $SERVICE_NAME"
echo ""
echo "  3. Проверь логи:"
echo "       journalctl -u $SERVICE_NAME -f"
echo "═══════════════════════════════════════════════════════"
