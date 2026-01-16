import shutil
from django.conf import settings
import rust_backend


def save_and_backup_face_db():
    """
    Save Rust DB (debounced internally) and rotate Python backups.
    """
    base_path = str(settings.FACE_DATABASE_PATH)
    backups_path = settings.FACE_DB_BACKUPS

    backups_path.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Ask Rust to save (atomic + debounced)
    rust_backend.save_database(base_path)

    db_file = settings.FACE_DATABASE_PATH / "data.bin"
    if not db_file.exists():
        return

    # 2️⃣ Rotate backups (N → N-1)
    for i in range(settings.FACE_DB_MAX_BACKUPS, 0, -1):
        src = backups_path / f"data.bin.{i}"
        dst = backups_path / f"data.bin.{i + 1}"

        if src.exists():
            if i + 1 > settings.FACE_DB_MAX_BACKUPS:
                src.unlink()
            else:
                src.rename(dst)

    # 3️⃣ Copy latest
    shutil.copy2(db_file, backups_path / "data.bin.1")
