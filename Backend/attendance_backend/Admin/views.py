from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils.timezone import now
import hashlib, jwt, json
from Admin.models import AdminAccessKey
from attendance.models import Organization

SECRET = "ADMIN_SECRET_2025"

def normalize_combo(combo: str) -> str:
    return "".join([c.upper() for c in combo.strip() if c != " "])

@csrf_exempt
def admin_login(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    body = json.loads(request.body)

    org_name = body.get("organization")
    entered = body.get("combo", "")

    if not org_name:
        return JsonResponse({"error": "Organization required"}, status=400)

    try:
        org = Organization.objects.get(name__iexact=org_name)
    except Organization.DoesNotExist:
        return JsonResponse({"error": "Organization not found"}, status=404)

    hashed = hashlib.sha256(normalize_combo(entered).encode()).hexdigest()

    key = AdminAccessKey.objects.filter(
        organization=org,
        active=True
    ).order_by("-created_at").first()

    if not key or key.combo_hash != hashed:
        return JsonResponse({"success": False, "message": "Invalid credentials"}, status=401)

    token = jwt.encode(
        {
            "admin": True,
            "org_id": org.id,
            "org_name": org.name,
            "time": str(now())
        },
        SECRET,
        algorithm="HS256"
    )

    return JsonResponse({
        "success": True,
        "token": token,
        "organization": org.name
    })
