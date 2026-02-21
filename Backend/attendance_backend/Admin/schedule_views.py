import json
import logging
from datetime import datetime

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.db import transaction

from attendance.models import (
    Teacher,
    Timetable,
    Branch,
)

from .face_security import admin_required
import intelligence_py

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# MARK TEACHER ABSENT + GENERATE SUBSTITUTION
# -------------------------------------------------------

@csrf_exempt
@admin_required
@require_http_methods(["POST"])
def generate_substitution(request):
    """
    Runtime substitution engine.

    Input:
    {
        "teacher_employee_id": "T001",
        "date": "2026-02-21"
    }

    Output:
    {
        substitutions: [...]
    }
    """

    org_id = request.admin["org_id"]

    try:
        data = json.loads(request.body)
        teacher_id = data["teacher_employee_id"]
        date_str = data["date"]
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({"error": "Invalid date format. Use YYYY-MM-DD"}, status=400)

    day_of_week = target_date.strftime("%A")

    # ---------------- FIND TEACHER ----------------
    try:
        teacher = Teacher.objects.select_related("branch").get(
            employee_id=teacher_id,
            branch__organization_id=org_id
        )
    except Teacher.DoesNotExist:
        return JsonResponse({"error": "Teacher not found"}, status=404)

    branch = teacher.branch

    # ---------------- GET TODAY'S CLASSES ----------------
    todays_classes = Timetable.objects.filter(
        teacher=teacher,
        day_of_week=day_of_week
    ).select_related("school_class")

    if not todays_classes.exists():
        return JsonResponse({
            "message": "No scheduled classes for this teacher on that day.",
            "substitutions": []
        })

    # ---------------- BUILD RUST PAYLOAD ----------------
    rust_payload = []

    for entry in todays_classes:
        rust_payload.append({
            "class_name": entry.school_class.class_name,
            "section": entry.school_class.section,
            "subject": entry.subject,
            "start_time": str(entry.start_time),
            "end_time": str(entry.end_time),
            "absent_teacher": teacher.employee_id,
        })

    # ---------------- CALL RUST ENGINE ----------------
    try:
        substitutions = intelligence_py.schedule_classes(
            rust_payload,
            str(org_id)
        )
    except Exception:
        logger.exception("Rust substitution engine failed")
        return JsonResponse({"error": "Scheduling engine failed"}, status=500)

    return JsonResponse({
        "date": str(target_date),
        "day": day_of_week,
        "absent_teacher": teacher.employee_id,
        "substitutions": substitutions
    })