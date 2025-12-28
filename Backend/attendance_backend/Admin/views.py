from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.utils.timezone import now
from django.conf import settings
from datetime import timedelta
import hashlib
import jwt
import json
import hmac
from django.core.cache import cache
from Admin.models import AdminAccessKey
from attendance.models import Branch, Organization, Teacher
from .utils import admin_required

SECRET = settings.ADMIN_JWT_SECRET


def normalize_combo(combo: str) -> str:
    return "".join(c.upper() for c in combo if c != " ")


def is_rate_limited(ip, org_id):
    key = f"admin_login:{ip}:{org_id}"
    attempts = cache.get(key, 0)
    if attempts >= 5:
        return True
    cache.set(key, attempts + 1, timeout=300)
    return False

# Admin Auth Views
@csrf_exempt
def admin_login(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    org_name = body.get("organization", "").strip()
    entered = body.get("combo", "")

    if not org_name or not entered:
        return JsonResponse({"error": "Invalid credentials"}, status=401)

    if len(entered) < 6 or len(entered) > 64:
        return JsonResponse({"error": "Invalid credentials"}, status=401)

    try:
        org = Organization.objects.get(name__iexact=org_name)
    except Organization.DoesNotExist:
        return JsonResponse({"error": "Invalid credentials"}, status=401)

    ip = request.META.get("REMOTE_ADDR", "unknown")
    if is_rate_limited(ip, org.id):
        return JsonResponse({"error": "Too many attempts. Try later."}, status=429)

    hashed = hashlib.sha256(normalize_combo(entered).encode()).hexdigest()

    key = (
        AdminAccessKey.objects
        .filter(organization=org, active=True)
        .order_by("-created_at")
        .first()
    )

    if not key or not hmac.compare_digest(key.combo_hash, hashed):
        return JsonResponse({"error": "Invalid credentials"}, status=401)

    token = jwt.encode(
        {
            "admin": True,
            "org_id": org.id,
            "org_name": org.name,
            "iat": int(now().timestamp()),
            "exp": int((now() + timedelta(minutes=30)).timestamp()),
            "iss": "attendance-backend",
            "aud": "attendance-admin",
        },
        SECRET,
        algorithm="HS256",
    )

    return JsonResponse({
        "success": True,
        "token": token,
        "organization": org.name
    })

# Organization Views

@admin_required
@require_http_methods(["GET"])
def get_organization(request):
    org = Organization.objects.get(id=request.admin["org_id"])

    return JsonResponse({
        "name": org.name,
        "org_type": org.org_type,
        "website": org.website,
        "created_at": org.created_at,
    })


@admin_required
@require_http_methods(["PUT"])
def update_organization(request):
    org = Organization.objects.get(id=request.admin["org_id"])

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Controlled updates
    org.org_type = body.get("org_type", org.org_type)
    org.website = body.get("website", org.website)

    # Optional: lock rename later
    # org.name = body.get("name", org.name)

    org.save()

    return JsonResponse({"success": True})


@admin_required
@require_http_methods(["POST"])
def rotate_admin_combo(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    new_combo = body.get("combo", "")

    if len(new_combo) < 6 or len(new_combo) > 64:
        return JsonResponse({"error": "Weak combo"}, status=400)

    org = Organization.objects.get(id=request.admin["org_id"])

    # Disable old keys
    AdminAccessKey.objects.filter(
        organization=org,
        active=True
    ).update(active=False)

    hashed = hashlib.sha256(
        normalize_combo(new_combo).encode()
    ).hexdigest()

    AdminAccessKey.objects.create(
        organization=org,
        combo_hash=hashed,
        active=True,
        is_default=False
    )

    return JsonResponse({"success": True})


# Branch Views

@csrf_exempt
@admin_required
@require_http_methods(["GET", "POST"])
def branches_view(request):
    org_id = request.admin["org_id"]
    organization = Organization.objects.get(id=org_id)

    # ================= LIST =================
    if request.method == "GET":
        branches = Branch.objects.filter(
            organization=organization).order_by("name")

        return JsonResponse([
            {
                "id": b.id,
                "organization": b.organization.id,
                "organization_name": b.organization.name,
                "name": b.name,
                "address": b.address,
                "city": b.city,
                "state": b.state,
                "pincode": b.pincode,
                "latitude": b.latitude,
                "longitude": b.longitude,
                "attendance_radius": b.attendance_radius,
            }
            for b in branches
        ], safe=False)

    # ================= CREATE =================
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = body.get("name", "").strip()

    if not name:
        return JsonResponse({"error": "Branch name required"}, status=400)

    branch = Branch.objects.create(
        organization=organization,
        name=name,
        address=body.get("address"),
        city=body.get("city"),
        state=body.get("state"),
        pincode=body.get("pincode"),
        latitude=body.get("latitude"),
        longitude=body.get("longitude"),
        attendance_radius=body.get("attendance_radius", 75),
    )

    return JsonResponse({
        "success": True,
        "id": branch.id
    }, status=201)


@csrf_exempt
@admin_required
@require_http_methods(["GET", "PUT", "DELETE"])
def branch_detail_view(request, branch_id):
    org_id = request.admin["org_id"]

    try:
        branch = Branch.objects.get(id=branch_id, organization_id=org_id)
    except Branch.DoesNotExist:
        return JsonResponse({"error": "Branch not found"}, status=404)

    # ================= READ =================
    if request.method == "GET":
        return JsonResponse({
            "id": branch.id,
            "name": branch.name,
            "address": branch.address,
            "city": branch.city,
            "state": branch.state,
            "pincode": branch.pincode,
            "latitude": branch.latitude,
            "longitude": branch.longitude,
            "attendance_radius": branch.attendance_radius,
        })

    # ================= UPDATE =================
    if request.method == "PUT":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        branch.name = body.get("name", branch.name)
        branch.address = body.get("address", branch.address)
        branch.city = body.get("city", branch.city)
        branch.state = body.get("state", branch.state)
        branch.pincode = body.get("pincode", branch.pincode)
        branch.latitude = body.get("latitude", branch.latitude)
        branch.longitude = body.get("longitude", branch.longitude)
        branch.attendance_radius = body.get(
            "attendance_radius", branch.attendance_radius
        )

        branch.save()

        return JsonResponse({"success": True})

    # ================= DELETE =================
    branch.delete()
    return JsonResponse({"success": True})

# Teacher Views

@csrf_exempt
@admin_required
@require_http_methods(["GET", "POST"])
def teachers_view(request):
    org_id = request.admin["org_id"]

    # ================= LIST =================
    if request.method == "GET":
        teachers = Teacher.objects.filter(
            branch__organization_id=org_id
        ).select_related("branch", "department")

        return JsonResponse([
            {
                "id": t.id,
                "name": t.name,
                "employee_id": t.employee_id,
                "branch": t.branch.id,
                "branch_name": t.branch.name,
                "department": t.department.name if t.department else None,
                "subjects": t.subjects,
                "can_teach_classes": t.can_teach_classes,
                "reliability_score": t.reliability_score,
            }
            for t in teachers
        ], safe=False)

    # ================= CREATE =================
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = body.get("name", "").strip()
    employee_id = body.get("employee_id", "").strip()
    branch_id = body.get("branch")

    if not all([name, employee_id, branch_id]):
        return JsonResponse({"error": "Missing required fields"}, status=400)

    try:
        branch = Branch.objects.get(
            id=branch_id,
            organization_id=org_id
        )
    except Branch.DoesNotExist:
        return JsonResponse({"error": "Invalid branch"}, status=403)

    teacher = Teacher.objects.create(
        name=name,
        employee_id=employee_id,
        branch=branch,
        department_id=body.get("department"),
        subjects=body.get("subjects", []),
        can_teach_classes=body.get("can_teach_classes", []),
    )

    return JsonResponse({"success": True, "id": teacher.id}, status=201)

@csrf_exempt
@admin_required
@require_http_methods(["GET", "PUT", "DELETE"])
def teacher_detail_view(request, teacher_id):
    org_id = request.admin["org_id"]

    try:
        teacher = Teacher.objects.select_related("branch").get(
            id=teacher_id,
            branch__organization_id=org_id
        )
    except Teacher.DoesNotExist:
        return JsonResponse({"error": "Teacher not found"}, status=404)

    # ================= READ =================
    if request.method == "GET":
        return JsonResponse({
            "id": teacher.id,
            "name": teacher.name,
            "employee_id": teacher.employee_id,
            "branch": teacher.branch.id,
            "department": teacher.department_id,
            "subjects": teacher.subjects,
            "can_teach_classes": teacher.can_teach_classes,
            "reliability_score": teacher.reliability_score,
            "workload_score": teacher.workload_score,
        })

    # ================= UPDATE =================
    if request.method == "PUT":
        body = json.loads(request.body)

        teacher.name = body.get("name", teacher.name)
        teacher.subjects = body.get("subjects", teacher.subjects)
        teacher.can_teach_classes = body.get("can_teach_classes", teacher.can_teach_classes)
        teacher.reliability_score = body.get("reliability_score", teacher.reliability_score)
        teacher.workload_score = body.get("workload_score", teacher.workload_score)

        teacher.save()
        return JsonResponse({"success": True})

    # ================= DELETE =================
    teacher.delete()
    return JsonResponse({"success": True})
