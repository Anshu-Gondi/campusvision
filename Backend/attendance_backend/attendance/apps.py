# attendance/apps.py
from django.apps import AppConfig
from django.conf import settings
import shutil
import requests
import os
import time

AXUM_SERVER_URL = "http://127.0.0.1:3000"


def wait_for_axum(timeout=10):
    for _ in range(timeout):
        try:
            requests.get(f"{AXUM_SERVER_URL}/health", timeout=1)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "attendance"

    def ready(self):
        # 🔒 Prevent double-run due to Django autoreload
        if os.environ.get("RUN_MAIN") != "true":
            return

        # 🚦 Ensure Axum is up before touching DB
        if not wait_for_axum():
            print("❌ Axum not available, skipping face DB initialization")
            return

        base_path = settings.FACE_DATABASE_PATH
        db_file = base_path / "data.bin"
        backups_path = settings.FACE_DB_BACKUPS

        base_path.mkdir(parents=True, exist_ok=True)
        backups_path.mkdir(parents=True, exist_ok=True)

        # 1️⃣ Try normal load
        try:
            resp = requests.post(
                f"{AXUM_SERVER_URL}/face/load",
                timeout=5,
            )
            resp.raise_for_status()
            print("✅ Face database loaded via Axum")
            return
        except requests.RequestException as e:
            print(f"⚠️ Failed to load face DB via Axum: {e}")

        # 2️⃣ Restore from backups
        for i in range(1, settings.FACE_DB_MAX_BACKUPS + 1):
            backup = backups_path / f"data.bin.{i}"
            if backup.exists():
                try:
                    shutil.copy2(backup, db_file)
                    resp = requests.post(
                        f"{AXUM_SERVER_URL}/face/load",
                        timeout=5,
                    )
                    resp.raise_for_status()
                    print(f"♻️ Restored face DB from backup data.bin.{i}")
                    return
                except requests.RequestException as e:
                    print(f"⚠️ Backup {i} failed: {e}")

        # 3️⃣ Initialize empty DB
        try:
            resp = requests.post(
                f"{AXUM_SERVER_URL}/face/init",
                timeout=5,
            )
            resp.raise_for_status()
            print("⚠️ No valid DB found — initialized empty DB via Axum")
        except requests.RequestException as e:
            print(f"❌ Failed to initialize empty DB via Axum: {e}")
