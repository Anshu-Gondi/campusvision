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
