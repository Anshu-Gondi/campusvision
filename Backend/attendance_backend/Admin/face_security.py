import jwt
from django.conf import settings
from django.http import JsonResponse
from attendance.models import FaceRejectionLog
from .utils import rust_client as rust_backend
from PIL import Image
import io

SECRET = settings.ADMIN_JWT_SECRET
RUST_BASE_URL = getattr(settings, "RUST_BASE_URL", "http://127.0.0.1:3000")
RUST_TIMEOUT = 10


def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        auth = request.headers.get("Authorization")

        if not auth or not auth.startswith("Bearer "):
            return JsonResponse({"error": "Unauthorized"}, status=401)

        token = auth.split(" ")[1]

        try:
            payload = jwt.decode(
                token,
                SECRET,
                algorithms=["HS256"],
                issuer="attendance-backend",
                audience="attendance-admin",
            )
        except jwt.ExpiredSignatureError:
            return JsonResponse({"error": "Session expired"}, status=401)
        except jwt.InvalidTokenError:
            return JsonResponse({"error": "Invalid token"}, status=401)

        # attach payload for later use
        request.admin = payload
        return view_func(request, *args, **kwargs)

    return wrapper


def safe_admin_id(admin):
    if not admin:
        return None
    return admin.get("admin_id") or admin.get("sub")


# Face recognition thresholds
DUPLICATE_THRESHOLD = 0.75   # reject other person's face
REENROLL_THRESHOLD = 0.65    # allow same person update
QUALITY_THRESHOLD = 0.55     # blur / side face
MAX_IMAGES_PER_PERSON = 5           # future-proof


# Face rejection logging
def log_face_rejection(
    *,
    reason,
    role,
    admin,
    person_id=None,
    similarity=None,
    threshold=None,
    message=""
):
    FaceRejectionLog.objects.create(
        reason=reason,
        role=role,
        person_id=person_id,
        admin_id=safe_admin_id(admin),
        similarity=similarity,
        threshold=threshold,
        message=message,
    )


def process_face_upload(file, role, admin=None, person_id=None):
    image_bytes = file.read()

    # Image metadata (huge signal)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        image_info = f"{img.format} {img.width}x{img.height}"
    except Exception:
        image_info = "unreadable"

    # ── Detect + Embed (Rust) ───────────────────────────────
    try:
        result = rust_backend.detect_and_embed(image_bytes)

    except rust_backend.RustAPIError as e:
        reason = (
            "rust_client_error"
            if e.status and e.status < 500
            else "rust_server_error"
        )

        log_face_rejection(
            reason=reason,
            role=role,
            admin=admin,
            person_id=person_id,
            message=(
                f"status={e.status} "
                f"error={e.error} "
                f"raw={e.raw} "
                f"image_bytes={len(image_bytes)} "
                f"image_info={image_info}"
            ),
        )
        return None, None  # ⛔ HARD STOP

    except Exception as e:
        log_face_rejection(
            reason="unexpected_exception",
            role=role,
            admin=admin,
            person_id=person_id,
            message=f"{type(e).__name__}: {str(e)}",
        )
        return None, None

    return result, image_info