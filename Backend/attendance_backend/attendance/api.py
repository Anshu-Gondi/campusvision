from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils.timezone import now
import hashlib
import jwt

SECRET = "ADMIN_SECRET_2025"

@csrf_exempt
def admin_login(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    import json
    body = json.loads(request.body)
    entered = body.get("combo")

    hashed = hashlib.sha256(entered.encode()).hexdigest()

    from attendance.models import AdminAccessKey
    key = AdminAccessKey.objects.first()

    if key is None:
        return JsonResponse({"error": "No admin key found"}, status=404)

    if hashed != key.combo_hash:
        return JsonResponse({"success": False, "message": "Invalid combo"}, status=401)

    token = jwt.encode({"admin": True, "time": str(now())}, SECRET, algorithm="HS256")

    return JsonResponse({"success": True, "token": token})
