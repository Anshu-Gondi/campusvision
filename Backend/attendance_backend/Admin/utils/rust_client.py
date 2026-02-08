# Admin/utils/rust_client.py
import requests
from django.conf import settings

RUST_BASE = getattr(settings, "RUST_BASE_URL", "http://127.0.0.1:3000")


def detect_and_embed(image_bytes):
    resp = requests.post(
        f"{RUST_BASE}/face/detect-embed",
        json={"image": list(image_bytes)},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def add_person(embedding, name, person_id, roll_no, role):
    resp = requests.post(
        f"{RUST_BASE}/face/add-person",
        json={
            "embedding": embedding,
            "name": name,
            "person_id": person_id,
            "roll_no": roll_no,
            "role": role,
        },
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()["id"]


def search_person(embedding, role, k=1):
    resp = requests.post(
        f"{RUST_BASE}/face/search",
        json={"embedding": embedding, "role": role, "k": k},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()["results"]


def count_students():
    resp = requests.get(f"{RUST_BASE}/face/count?role=student", timeout=10)
    resp.raise_for_status()
    return resp.json()["count"]


def count_teachers():
    resp = requests.get(f"{RUST_BASE}/face/count?role=teacher", timeout=10)
    resp.raise_for_status()
    return resp.json()["count"]


def save_database():
    resp = requests.post(f"{RUST_BASE}/face/save", timeout=10)
    resp.raise_for_status()
    return resp.json()
