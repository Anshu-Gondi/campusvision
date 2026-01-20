import jwt
from django.conf import settings
from django.http import JsonResponse
from attendance.models import FaceRejectionLog
import rust_backend

SECRET = settings.ADMIN_JWT_SECRET


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

    # ---- Detect + Embed (Rust) ----
    try:
        result = rust_backend.detect_and_embed(
            image_bytes,
            settings.YUNET_MODEL_PATH,
        )

        if not settings.YUNET_MODEL_PATH:
            raise RuntimeError("YuNet model path is not configured")

    except Exception as e:
        log_face_rejection(
            reason="rust_failure",
            role=role,
            admin=admin,
            person_id=person_id,
            message=str(e),
        )
        return None, None   # 🔴 HARD STOP

    # ---- NO FACE FOUND ----
    if not result or not result.get("found"):
        log_face_rejection(
            reason="no_face_detected",
            role=role,
            admin=admin,
            person_id=person_id,
            message="No face detected in image",
        )
        return None, None   # 🔴 HARD STOP

    embedding = result["embedding"]

    # ---- QUALITY GATE ----
    total = (
        rust_backend.count_students()
        if role == "student"
        else rust_backend.count_teachers()
    )

    if total > 0:
        matches = rust_backend.search_person(embedding, role, 1)
        if matches:
            _, max_sim = matches[0]
            if max_sim < QUALITY_THRESHOLD:
                log_face_rejection(
                    reason="low_quality",
                    role=role,
                    admin=admin,
                    person_id=person_id,
                    similarity=max_sim,
                    threshold=QUALITY_THRESHOLD,
                    message="Blurred / side-face / low confidence",
                )
                return None, None   # 🔴 HARD STOP

    return embedding, image_bytes
