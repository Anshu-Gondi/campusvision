# attendance/utils/face_db_backup.py
import shutil
import requests
import time
from django.conf import settings

AXUM_SERVER_URL = "http://127.0.0.1:3000"


def wait_for_axum(timeout=5):
    for _ in range(timeout):
        try:
            requests.get(f"{AXUM_SERVER_URL}/health", timeout=1)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


def save_and_backup_face_db():
    """
    Save face database via Axum API and rotate local backups.
    """

    # 🚦 Ensure Axum is available
    if not wait_for_axum():
        print("❌ Axum not available — skipping face DB save")
        return

    base_path = settings.FACE_DATABASE_PATH
    backups_path = settings.FACE_DB_BACKUPS
    db_file = base_path / "data.bin"

    backups_path.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Ask Axum to save
    try:
        resp = requests.post(
            f"{AXUM_SERVER_URL}/face/save",
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Failed to save face DB via Axum: {e}")
        return

    # 2️⃣ Verify DB exists
    if not db_file.exists():
        print("❌ DB file not found after Axum save")
        return

    # 3️⃣ Rotate backups (highest → lowest)
    for i in range(settings.FACE_DB_MAX_BACKUPS, 0, -1):
        src = backups_path / f"data.bin.{i}"
        dst = backups_path / f"data.bin.{i + 1}"

        try:
            if src.exists():
                if i + 1 > settings.FACE_DB_MAX_BACKUPS:
                    src.unlink()
                else:
                    src.replace(dst)
        except Exception as e:
            print(f"⚠️ Failed rotating backup {i}: {e}")
            return

    # 4️⃣ Write latest backup
    try:
        shutil.copy2(db_file, backups_path / "data.bin.1")
        print("✅ Face DB saved and backup rotated")
    except Exception as e:
        print(f"❌ Failed to write backup: {e}")
