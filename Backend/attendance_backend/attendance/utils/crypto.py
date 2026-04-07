from cryptography.fernet import Fernet
from django.conf import settings

# store this key in settings.py
fernet = Fernet(settings.SECRET_ENCRYPTION_KEY)

def encrypt_value(value: str) -> str:
    return fernet.encrypt(value.encode()).decode()

def decrypt_value(value: str) -> str:
    return fernet.decrypt(value.encode()).decode()