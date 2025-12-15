import os
import django
import hashlib

# IMPORTANT: point to your Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_backend.settings")

django.setup()
# Admin/models.py already has AdminAccessKey
import hashlib
from Admin.models import AdminAccessKey

# Your secret key (no spaces)
key = "C3C3↑"

def normalize_combo(combo: str) -> str:
    """
    Normalize combo:
    - Uppercase letters
    - Keep arrow symbols
    - Ignore spaces completely
    """
    return "".join([c.upper() for c in combo.strip() if c != " "])

# Hash the normalized combo
hashed = hashlib.sha256(normalize_combo(key).encode()).hexdigest()

# Clear old keys
AdminAccessKey.objects.all().delete()

# Save new key
AdminAccessKey.objects.create(combo_hash=hashed)
print("Admin key created successfully!")
