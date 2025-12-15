import hashlib, secrets
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from attendance.models import Organization, Branch
from Admin.models import AdminAccessKey

@csrf_exempt
def register_organization(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    import json
    data = json.loads(request.body)

    # 1️⃣ Create organization
    org = Organization.objects.create(
        name=data["org_name"],
        org_type=data["org_type"],
        website=data.get("website")
    )

    # 2️⃣ Create branch
    branch = Branch.objects.create(
        organization=org,
        name=data["branch_name"],
        city=data.get("city"),
        state=data.get("state")
    )

    # 3️⃣ Generate default admin password
    raw_password = secrets.token_hex(4).upper()  # e.g. A9F3D2C1
    hashed = hashlib.sha256(raw_password.encode()).hexdigest()

    AdminAccessKey.objects.create(
        organization=org,
        combo_hash=hashed,
        is_default=True
    )

    return JsonResponse({
        "success": True,
        "org_id": org.id,
        "default_admin_password": raw_password
    })
