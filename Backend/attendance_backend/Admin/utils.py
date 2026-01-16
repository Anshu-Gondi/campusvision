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
        admin_id=admin.get("sub") or admin.get("admin_id"),
        similarity=similarity,
        threshold=threshold,
        message=message,
    )

def process_face_upload(file, role, admin=None, person_id=None):
    image_bytes = file.read()

    # ---- Detect + Embed (Rust) ----
    try:
        result = rust_backend.detect_and_embed(image_bytes, settings.YUNET_MODEL_PATH,)
    except Exception as e:
        if admin:
            FaceRejectionLog.objects.create(
                reason="rust_failure",
                role=role,
                admin_id=admin.get("sub") or admin.get("admin_id"),
                person_id=person_id,
                message=str(e),
            )
        raise ValueError("Face processing failed")

    if not result.get("found"):
        if admin:
            FaceRejectionLog.objects.create(
                reason="no_face_detected",
                role=role,
                admin_id=admin.get("sub") or admin.get("admin_id"),
                person_id=person_id,
                message="No face detected in image",
            )
        raise ValueError("No face detected")

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
            if max_sim < 0.55:  # QUALITY_THRESHOLD
                if admin:
                    FaceRejectionLog.objects.create(
                        reason="low_quality",
                        role=role,
                        admin_id=admin.get("sub") or admin.get("admin_id"),
                        person_id=person_id,
                        similarity=max_sim,
                        threshold=0.55,
                        message="Blurred / side-face / low confidence",
                    )
                raise ValueError("Low-quality face. Please retake image.")

    return embedding, image_bytes
