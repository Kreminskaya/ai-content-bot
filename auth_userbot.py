# -*- coding: utf-8 -*-
"""
One-time Telethon authorisation script.
Run this ONCE before starting the bot for the first time:

    python auth_userbot.py

What it does:
  1. Connects to Telegram using your API credentials from .env
  2. Sends an SMS / Telegram code to TELEGRAM_PHONE
  3. Asks you to enter the code (and 2FA password if enabled)
  4. Saves the session to USERBOT_SESSION (default: userbot.session)

After that the bot uses the saved session automatically -- no repeated auth needed.

If you ever need to re-authorise (e.g. session expired or revoked),
just delete userbot.session and run this script again.
"""

import asyncio
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows so the script prints cleanly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from config import (
    TELEGRAM_API_HASH,
    TELEGRAM_API_ID,
    TELEGRAM_PHONE,
    USERBOT_SESSION,
)


def _print(msg: str) -> None:
    """Print with flush so output appears immediately even in some terminals."""
    print(msg, flush=True)


async def main() -> None:
    # -- Validate credentials ------------------------------------------------
    missing = []
    if not TELEGRAM_API_ID:
        missing.append("TELEGRAM_API_ID")
    if not TELEGRAM_API_HASH:
        missing.append("TELEGRAM_API_HASH")
    if not TELEGRAM_PHONE:
        missing.append("TELEGRAM_PHONE")

    if missing:
        _print(
            "\n[ERROR] Missing credentials in .env:\n"
            + "\n".join(f"   - {v}" for v in missing)
            + "\n\nFill them in and re-run this script.\n"
        )
        return

    # -- Import Telethon after config is validated ---------------------------
    try:
        from telethon import TelegramClient
        from telethon.errors import SessionPasswordNeededError
    except ImportError:
        _print(
            "\n[ERROR] telethon is not installed.\n"
            "    Run:  pip install telethon\n"
        )
        return

    api_id = int(TELEGRAM_API_ID)

    _print(f"\n[*] Connecting to Telegram as {TELEGRAM_PHONE} ...")
    _print(f"    Session will be saved to: {USERBOT_SESSION}\n")

    client = TelegramClient(USERBOT_SESSION, api_id, TELEGRAM_API_HASH)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        _print(f"[OK] Already authorised as @{me.username} ({me.first_name}).")
        _print("     Nothing to do -- the session is valid.\n")
        await client.disconnect()
        return

    # -- Send the code -------------------------------------------------------
    try:
        await client.send_code_request(TELEGRAM_PHONE)
    except Exception as exc:
        _print(f"\n[ERROR] Could not send code: {exc}\n")
        await client.disconnect()
        return

    _print("[*] A confirmation code was sent to your Telegram app (or via SMS).")
    code = input("    Enter the code: ").strip()

    try:
        await client.sign_in(TELEGRAM_PHONE, code)

    except SessionPasswordNeededError:
        # Two-step verification is enabled on this account
        _print("\n[*] Two-step verification is enabled on this account.")
        password = input("    Enter your 2FA password: ").strip()
        try:
            await client.sign_in(password=password)
        except Exception as exc:
            _print(f"\n[ERROR] 2FA sign-in failed: {exc}\n")
            await client.disconnect()
            return

    except Exception as exc:
        _print(f"\n[ERROR] Sign-in failed: {exc}\n")
        await client.disconnect()
        return

    # -- Success -------------------------------------------------------------
    me = await client.get_me()
    _print(
        f"\n[OK] Authorised as @{me.username} ({me.first_name})!\n"
        f"     Session saved to: {USERBOT_SESSION}\n"
        "\n"
        "     You can now start the bot:\n"
        "         python main.py\n"
    )
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
