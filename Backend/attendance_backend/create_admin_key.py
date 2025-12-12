import os
import django
import hashlib

# IMPORTANT: this MUST point to your project settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_backend.settings")

django.setup()

from attendance.models import AdminAccessKey

key = "C 3 C3↑"
hashed = hashlib.sha256(key.encode()).hexdigest()

AdminAccessKey.objects.create(combo_hash=hashed)
print("Admin key created successfully!")
