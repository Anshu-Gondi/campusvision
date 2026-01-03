import jwt
from django.conf import settings
from django.http import JsonResponse

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

import rust_backend

DUPLICATE_REJECT_THRESHOLD = 0.75   # same person → reject
MIN_QUALITY_SIMILARITY = 0.55       # blurry / bad face → reject
MAX_IMAGES_PER_PERSON = 5           # future-proof


def process_face_upload(file, role):
    """
    1. Detect face
    2. Extract embedding
    3. Quality gate
    4. Duplicate check
    """
    image_bytes = file.read()

    result = rust_backend.detect_and_embed(image_bytes)

    if not result["found"]:
        raise ValueError("No face detected")

    embedding = result["embedding"]

    # ---- Duplicate check ----
    dup = rust_backend.check_duplicate(
        embedding,
        role,
        DUPLICATE_REJECT_THRESHOLD
    )

    if dup.get("duplicate"):
        raise ValueError(
            f"Duplicate face detected: {dup['name']} "
            f"(similarity={dup['similarity']:.2f})"
        )

    # ---- Quality gate (nearest neighbor similarity) ----
    nearest = rust_backend.search_person(embedding, role, 1)
    if nearest and nearest[0][1] < MIN_QUALITY_SIMILARITY:
        raise ValueError("Low-quality face. Please retake the image.")

    return embedding, image_bytes