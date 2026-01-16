from django.apps import AppConfig
from django.conf import settings
import shutil


class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "attendance"

    def ready(self):
        """
        Runs when Django starts.
        Loads Rust face database with automatic recovery.
        """
        try:
            import rust_backend
        except ImportError:
            print("⚠️ Rust backend not installed yet.")
            return

        base_path = settings.FACE_DATABASE_PATH
        db_file = base_path / "data.bin"
        backups_path = settings.FACE_DB_BACKUPS

        # 1️⃣ Try normal load
        try:
            rust_backend.load_database(str(base_path))
            print(f"✅ Rust face database loaded from: {base_path}")
            return
        except Exception as e:
            print(f"⚠️ Failed to load face DB: {e}")

        # 2️⃣ Try restore from backups
        for i in range(1, settings.FACE_DB_MAX_BACKUPS + 1):
            backup = backups_path / f"data.bin.{i}"
            if backup.exists():
                try:
                    shutil.copy2(backup, db_file)
                    rust_backend.load_database(str(base_path))
                    print(f"♻️ Restored face DB from backup: data.bin.{i}")
                    return
                except Exception as e:
                    print(f"⚠️ Backup {i} failed: {e}")

        # 3️⃣ No DB, no backup → start empty
        print("⚠️ No valid face DB found. Starting with empty database.")
        base_path.mkdir(parents=True, exist_ok=True)
