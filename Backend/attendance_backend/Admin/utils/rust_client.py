# admin/utils/rust_client.py
import requests
from django.conf import settings

RUST_BASE = getattr(settings, "RUST_BASE_URL", "http://127.0.0.1:3000")
TIMEOUT = 180


class RustAPIError(Exception):
    def __init__(self, *, status=None, error=None, raw=None):
        self.status = status
        self.error = error
        self.raw = raw
        super().__init__(
            f"RustAPIError status={status} error={error} raw={raw}"
        )


def _post_json(path, payload):
    try:
        resp = requests.post(
            f"{RUST_BASE}{path}",
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise RustAPIError(error="network_error", raw=str(e))

    if resp.status_code != 200:
        try:
            data = resp.json()
        except Exception:
            data = None

        raise RustAPIError(
            status=resp.status_code,
            error=data,
            raw=resp.text,
        )

    return resp.json()


# ── Face APIs ─────────────────────────────────────────────

def detect_and_embed(image_bytes):
    return _post_json(
        "/face/detect-embed",
        {
            "image": list(image_bytes),
            "model_path": None,
            "enrollment": True,
        },
    )


def add_person(embedding, name, person_id, roll_no, role):
    return _post_json(
        "/face/add-person",
        {
            "embedding": embedding,
            "name": name,
            "person_id": person_id,
            "roll_no": roll_no,
            "role": role,
        },
    )["id"]


def search_person(embedding, role, k=1):
    return _post_json(
        "/face/search",
        {"embedding": embedding, "role": role, "k": k},
    )["results"]


def count_students():
    return _post_json("/face/count?role=student", {})["count"]


def count_teachers():
    return _post_json("/face/count?role=teacher", {})["count"]


def save_database():
    return _post_json("/face/save", {})
