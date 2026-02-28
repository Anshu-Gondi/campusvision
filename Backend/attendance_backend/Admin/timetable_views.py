# timetable_views.py (production-grade optimized)

import io
import json
import logging
from io import BytesIO
from datetime import datetime, time
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponse
from django.db import transaction
from attendance.models import (
    Branch, Teacher, SchoolClass,
    Timetable
)
from .face_security import admin_required
from django.conf import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = [".xls", ".xlsx"]
VALID_DAYS = ["Monday", "Tuesday", "Wednesday",
              "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_EMPTY_DAY_START = getattr(
    settings, "DEFAULT_EMPTY_DAY_START", time(9, 0))
DEFAULT_EMPTY_DAY_END = getattr(settings, "DEFAULT_EMPTY_DAY_END", time(15, 0))


def parse_time_str(value):
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except Exception:
        raise ValueError("Time must be in HH:MM format")


def time_overlap(s1, e1, s2, e2):
    return max(s1, s2) < min(e1, e2)


def parse_time(value, row_idx, column_name):
    """Safely parse time from Excel, raise ValueError if invalid."""
    try:
        if pd.isna(value) or value == "":
            return None
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            return datetime.strptime(value.strip(), "%H:%M").time()
        return pd.to_datetime(value).time()
    except Exception:
        raise ValueError(
            f"Invalid {column_name} at row {row_idx + 2}: '{value}'")


@csrf_exempt
@admin_required
@require_http_methods(["POST"])
def upload_timetable_smart(request):
    org_id = request.admin["org_id"]
    preview_mode = request.GET.get("preview", "true").lower() == "true"

    if "file" not in request.FILES:
        return JsonResponse({"error": "Excel file required"}, status=400)

    file = request.FILES["file"]
    if not any(file.name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return JsonResponse({"error": "Invalid file format"}, status=400)

    try:
        df = pd.read_excel(BytesIO(file.read()))
    except Exception as e:
        logger.exception("Failed to read Excel file")
        return JsonResponse({"error": "Failed to read Excel file"}, status=400)

    required_columns = ["branch", "class_name", "section", "subject", "teacher_employee_id",
                        "day_of_week", "start_time", "end_time"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        return JsonResponse({"error": f"Missing columns: {missing_cols}"}, status=400)

    results = []
    timetable_rows = []

    # Prefetch branches and teachers to reduce DB queries
    branches_cache = {
        b.name.strip(): b for b in Branch.objects.filter(organization_id=org_id)}
    teachers_cache = {t.employee_id: t for t in Teacher.objects.filter(
        branch__organization_id=org_id)}

    for idx, row in df.iterrows():
        row_result = {"row": idx + 2}
        try:
            # ---------------- BRANCH ----------------
            branch_name = str(row["branch"]).strip()
            branch = branches_cache.get(branch_name)
            if not branch:
                raise ValueError(f"Branch '{branch_name}' not found")

            # ---------------- CLASS ----------------
            class_name = str(row["class_name"]).strip()
            section = str(row["section"]).strip()
            school_class, _ = SchoolClass.objects.get_or_create(
                branch=branch, class_name=class_name, section=section
            )

            # ---------------- SUBJECT & TEACHER ----------------
            subject = str(row["subject"]).strip(
            ) if pd.notna(row["subject"]) else ""
            teacher_id = str(row["teacher_employee_id"]).strip(
            ) if pd.notna(row["teacher_employee_id"]) else ""
            day_raw = str(row["day_of_week"]).strip(
            ) if pd.notna(row["day_of_week"]) else ""
            day = day_raw.title() if day_raw else None

            # ---------------- HANDLE EMPTY PERIOD ----------------
            start_time = parse_time(row.get("start_time"), idx, "start_time")
            end_time = parse_time(row.get("end_time"), idx, "end_time")

            if not subject or not day or not teacher_id:
                row_result["status"] = "skip"
                results.append(row_result)
                continue

            if day not in VALID_DAYS:
                raise ValueError(f"Invalid day_of_week: '{day}'")

            teacher = teachers_cache.get(teacher_id)
            if not teacher:
                raise ValueError(
                    f"Teacher '{teacher_id}' not found in branch '{branch_name}'")

            if not start_time or not end_time:
                raise ValueError(
                    "Both start_time and end_time must be provided for non-empty period")
            if start_time >= end_time:
                raise ValueError("start_time must be before end_time")

            # ---------------- CONFLICT CHECK ----------------
            if Timetable.objects.filter(
                teacher=teacher,
                day_of_week=day,
                start_time__lt=end_time,
                end_time__gt=start_time
            ).exists():
                raise ValueError(
                    f"Teacher '{teacher_id}' has overlapping class"
                )

            if Timetable.objects.filter(
                school_class=school_class,
                day_of_week=day,
                start_time__lt=end_time,
                end_time__gt=start_time
            ).exists():
                raise ValueError(
                    f"Class {class_name}-{section} has overlapping period"
                )

            # ---------------- PREPARE DB OBJECTS ----------------
            timetable_rows.append(Timetable(
                branch=branch,
                teacher=teacher,
                school_class=school_class,
                subject=subject,
                day_of_week=day,
                start_time=start_time,
                end_time=end_time
            ))

            row_result["status"] = "ok"

        except ValueError as ve:
            row_result["status"] = "fail"
            row_result["error"] = str(ve)
        except Exception as e:
            row_result["status"] = "fail"
            row_result["error"] = "Internal server error"
            logger.exception(f"Row {idx + 2} processing failed")

        results.append(row_result)

    # ---------------- PREVIEW MODE ----------------
    if preview_mode:
        return JsonResponse({
            "preview": True,
            "results": results,
            "success_count": sum(r["status"] == "ok" for r in results),
        })

    # ---------------- COMMIT TO DB ----------------
    try:
        with transaction.atomic():
            Timetable.objects.bulk_create(timetable_rows)
    except Exception as e:
        logger.exception("Failed to commit timetable or Rust scheduling")
        return JsonResponse({"error": "Failed to commit timetable"}, status=500)

    return JsonResponse({
        "preview": False,
        "results": results,
        "success_count": sum(r["status"] == "ok" for r in results)
    })


@csrf_exempt
@admin_required
@require_http_methods(["GET"])
def download_sample_timetable(request):
    """
    Returns a production-valid sample Excel using REAL DB data.
    """

    import random

    org_id = request.admin["org_id"]

    # ---------------- FETCH REAL DATA ----------------
    branches = list(
        Branch.objects.filter(organization_id=org_id)
    )

    if not branches:
        return JsonResponse({"error": "No branches found"}, status=400)

    teachers = list(
        Teacher.objects.filter(branch__organization_id=org_id)
        .select_related("branch")
    )

    if not teachers:
        return JsonResponse({"error": "No teachers found"}, status=400)

    # Group teachers by branch
    teachers_by_branch = {}
    for t in teachers:
        teachers_by_branch.setdefault(t.branch.id, []).append(t)

    # ---------------- SAMPLE CLASS STRUCTURE ----------------
    classes = [
        {"class_name": "10", "section": "A"},
        {"class_name": "10", "section": "B"},
        {"class_name": "11", "section": "A"},
    ]

    days_of_week = VALID_DAYS

    student_data = []
    teacher_data = []

    # ---------------- GENERATE SAMPLE ----------------
    for branch in branches:

        branch_teachers = teachers_by_branch.get(branch.id, [])
        if not branch_teachers:
            continue

        for cls in classes:
            class_name = cls["class_name"]
            section = cls["section"]

            for day in days_of_week:

                # Fully empty weekends
                if day in ["Saturday", "Sunday"]:
                    student_data.append({
                        "branch": branch.name,
                        "class_name": class_name,
                        "section": section,
                        "subject": "",
                        "teacher_employee_id": "",
                        "day_of_week": day,
                        "start_time": "",
                        "end_time": ""
                    })
                    continue

                num_periods = random.randint(2, 4)
                start_hour = 9

                for i in range(num_periods):

                    teacher = random.choice(branch_teachers)

                    subject = random.choice([
                        "Mathematics",
                        "Physics",
                        "Chemistry",
                        "English",
                        "Biology"
                    ])

                    start_time = f"{start_hour + i}:00"
                    end_time = f"{start_hour + i + 1}:00"

                    # Randomly create empty period
                    if random.random() < 0.2:
                        subject = ""
                        teacher_employee_id = ""
                    else:
                        teacher_employee_id = teacher.employee_id

                    student_data.append({
                        "branch": branch.name,
                        "class_name": class_name,
                        "section": section,
                        "subject": subject,
                        "teacher_employee_id": teacher_employee_id,
                        "day_of_week": day,
                        "start_time": start_time,
                        "end_time": end_time
                    })

    # ---------------- LEGEND ----------------
    legend_data = [
        {"Column": "branch", "Description": "Branch name (must match exactly)"},
        {"Column": "class_name", "Description": "Class number or name"},
        {"Column": "section", "Description": "Section of class"},
        {"Column": "subject", "Description": "Subject name"},
        {"Column": "teacher_employee_id", "Description": "REAL employee_id from DB"},
        {"Column": "day_of_week", "Description": f"Must be one of {VALID_DAYS}"},
        {"Column": "start_time", "Description": "HH:MM format"},
        {"Column": "end_time", "Description": "HH:MM format"},
    ]

    df_student = pd.DataFrame(student_data)
    df_legend = pd.DataFrame(legend_data)

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_student.to_excel(writer, index=False, sheet_name='Timetable')
        df_legend.to_excel(writer, index=False, sheet_name='Legend')

    buffer.seek(0)

    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    response['Content-Disposition'] = 'attachment; filename=sample_timetable.xlsx'
    return response


@csrf_exempt
@admin_required
@require_http_methods(["PUT"])
def update_timetable_entry(request, timetable_id):
    """
    Edit a single timetable entry safely (Unified Timetable Model).
    Optimized with DB-level overlap detection.
    """

    org_id = request.admin["org_id"]

    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    try:
        entry = Timetable.objects.select_related(
            "teacher", "school_class", "branch"
        ).get(id=timetable_id, branch__organization_id=org_id)

    except Timetable.DoesNotExist:
        return JsonResponse({"error": "Timetable entry not found"}, status=404)

    try:
        # --------- NEW VALUES ----------
        subject = data.get("subject", entry.subject)
        day = data.get("day_of_week", entry.day_of_week)

        start_time = (
            parse_time_str(data["start_time"])
            if "start_time" in data else entry.start_time
        )
        end_time = (
            parse_time_str(data["end_time"])
            if "end_time" in data else entry.end_time
        )

        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        teacher = entry.teacher
        school_class = entry.school_class

        # Overlap condition:
        # new_start < existing_end AND new_end > existing_start

        # --------- TEACHER CONFLICT CHECK ----------
        teacher_conflict_exists = Timetable.objects.filter(
            teacher=teacher,
            day_of_week=day,
            branch__organization_id=org_id
        ).exclude(id=entry.id).filter(
            start_time__lt=end_time,
            end_time__gt=start_time
        ).exists()

        if teacher_conflict_exists:
            raise ValueError("Teacher has overlapping period")

        # --------- CLASS CONFLICT CHECK ----------
        class_conflict_exists = Timetable.objects.filter(
            school_class=school_class,
            day_of_week=day,
            branch__organization_id=org_id
        ).exclude(id=entry.id).filter(
            start_time__lt=end_time,
            end_time__gt=start_time
        ).exists()

        if class_conflict_exists:
            raise ValueError("Class has overlapping period")

        # --------- ATOMIC UPDATE ----------
        with transaction.atomic():
            entry.subject = subject
            entry.day_of_week = day
            entry.start_time = start_time
            entry.end_time = end_time
            entry.save(update_fields=[
                "subject",
                "day_of_week",
                "start_time",
                "end_time"
            ])

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)

    except Exception:
        logger.exception("Timetable update failed")
        return JsonResponse({"error": "Internal server error"}, status=500)

    return JsonResponse({"success": True})


@csrf_exempt
@admin_required
@require_http_methods(["DELETE"])
def delete_timetable_entry(request, timetable_id):
    """
    Delete timetable entry safely (Unified Timetable Model).
    """

    org_id = request.admin["org_id"]

    try:
        entry = Timetable.objects.select_related(
            "school_class", "branch"
        ).get(id=timetable_id, branch__organization_id=org_id)

    except Timetable.DoesNotExist:
        return JsonResponse({"error": "Timetable entry not found"}, status=404)

    try:
        with transaction.atomic():
            school_class = entry.school_class
            entry.delete()

    except Exception:
        logger.exception("Timetable deletion failed")
        return JsonResponse({"error": "Failed to delete entry"}, status=500)

    return JsonResponse({"success": True})


@csrf_exempt
@admin_required
@require_http_methods(["GET"])
def list_timetable(request):
    """
    List timetable entries with filters.

    Query params (optional):
        ?branch_id=1
        ?teacher_id=5
        ?class_name=10
        ?section=A
        ?day=Monday
        ?page=1
        ?page_size=50
    """

    org_id = request.admin["org_id"]

    branch_id = request.GET.get("branch_id")
    teacher_id = request.GET.get("teacher_id")
    class_name = request.GET.get("class_name")
    section = request.GET.get("section")
    day = request.GET.get("day")

    page = int(request.GET.get("page", 1))
    page_size = min(int(request.GET.get("page_size", 50)), 200)

    queryset = Timetable.objects.select_related(
        "teacher", "school_class", "branch"
    ).filter(branch__organization_id=org_id)

    # ---------------- FILTERS ----------------
    if branch_id:
        queryset = queryset.filter(branch_id=branch_id)

    if teacher_id:
        queryset = queryset.filter(teacher_id=teacher_id)

    if class_name:
        queryset = queryset.filter(school_class__class_name=class_name)

    if section:
        queryset = queryset.filter(school_class__section=section)

    if day:
        queryset = queryset.filter(day_of_week=day)

    queryset = queryset.order_by(
        "day_of_week",
        "start_time"
    )

    total = queryset.count()

    # ---------------- PAGINATION ----------------
    start = (page - 1) * page_size
    end = start + page_size
    entries = queryset[start:end]

    data = []

    for entry in entries:
        data.append({
            "id": entry.id,
            "branch": entry.branch.name,
            "teacher_id": entry.teacher.id,
            "teacher_name": entry.teacher.name,
            "employee_id": entry.teacher.employee_id,
            "class_name": entry.school_class.class_name,
            "section": entry.school_class.section,
            "subject": entry.subject,
            "day_of_week": entry.day_of_week,
            "start_time": str(entry.start_time),
            "end_time": str(entry.end_time),
        })

    return JsonResponse({
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": data
    })
