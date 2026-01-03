from django.apps import AppConfig
from django.conf import settings


class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance'

    def ready(self):
        """
        This runs when Django starts.
        Loads the Rust face database if available.
        """
        try:
            from rust_backend import load_database
            load_database(str(settings.FACE_DATABASE_PATH))
            print(f"✅ Rust face database loaded from: {settings.FACE_DATABASE_PATH}")
        except ImportError:
            print("⚠️ Rust backend not installed yet.")
        except Exception as e:
            print(f"⚠️ Rust backend failed to load: {e}")
