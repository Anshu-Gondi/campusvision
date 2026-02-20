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
from .utils import rust_client as rust_backend
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
    rust_payload = []
    classes_days = {}

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

            key_cd = (branch.id, class_name, section)
            classes_days.setdefault(key_cd, set())
            if day:
                classes_days[key_cd].add(day)

            # ---------------- HANDLE EMPTY PERIOD ----------------
            start_time = parse_time(row.get("start_time"), idx, "start_time")
            end_time = parse_time(row.get("end_time"), idx, "end_time")

            if not subject or not day or not teacher_id:
                row_result["status"] = "empty"
                rust_payload.append({
                    "teacher_id": None,
                    "class_id": f"{class_name}-{section}",
                    "day": day or "Monday",
                    "start_time": str(start_time or DEFAULT_EMPTY_DAY_START),
                    "end_time": str(end_time or DEFAULT_EMPTY_DAY_END),
                    "subject": subject or "EMPTY",
                })
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
            teacher_conflicts = Timetable.objects.filter(
                teacher=teacher, day_of_week=day)
            for t in teacher_conflicts:
                if time_overlap(start_time, end_time, t.start_time, t.end_time):
                    raise ValueError(
                        f"Teacher '{teacher_id}' has overlapping class at {t.start_time}-{t.end_time}")

            class_conflicts = Timetable.objects.filter(
                school_class=school_class, day_of_week=day)
            for s in class_conflicts:
                if time_overlap(start_time, end_time, s.start_time, s.end_time):
                    raise ValueError(
                        f"Class {class_name}-{section} has overlapping period at {s.start_time}-{s.end_time}")

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

            # ---------------- RUST PAYLOAD ----------------
            rust_payload.append({
                "teacher_id": teacher.employee_id,
                "class_id": f"{class_name}-{section}",
                "day": day,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "subject": subject,
            })

            row_result["status"] = "ok"

        except ValueError as ve:
            row_result["status"] = "fail"
            row_result["error"] = str(ve)
        except Exception as e:
            row_result["status"] = "fail"
            row_result["error"] = "Internal server error"
            logger.exception(f"Row {idx + 2} processing failed")

        results.append(row_result)

    # ---------------- FULLY EMPTY DAYS ----------------
    for (branch_id, class_name, section), days_present in classes_days.items():
        for day in VALID_DAYS:
            if day not in days_present:
                rust_payload.append({
                    "teacher_id": None,
                    "class_id": f"{class_name}-{section}",
                    "day": day,
                    "start_time": str(DEFAULT_EMPTY_DAY_START),
                    "end_time": str(DEFAULT_EMPTY_DAY_END),
                    "subject": "EMPTY",
                })

    # ---------------- PREVIEW MODE ----------------
    if preview_mode:
        return JsonResponse({
            "preview": True,
            "results": results,
            "success_count": sum(r["status"] == "ok" for r in results),
            "rust_preview_payload": rust_payload
        })

    # ---------------- COMMIT TO DB ----------------
    try:
        with transaction.atomic():
            Timetable.objects.bulk_create(timetable_rows)
            rust_backend.schedule_classes(rust_payload)
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
    Returns a sample Excel showing both:
      - Teacher-side timetable
      - Student-side timetable
    Features:
      - Multiple classes & sections
      - Different teachers per section
      - Empty periods
      - Fully empty days
      - Legend sheet explaining how to fill the Excel
    """
    import random

    # ---------------- SCHOOL SETUP ----------------
    classes = [
        {"class_name": "10", "section": "A"},
        {"class_name": "10", "section": "B"},
        {"class_name": "11", "section": "A"},
    ]
    branch = "Main Campus"

    # Subject pool with teacher base ids
    subjects = [
        ("Mathematics", "T001"),
        ("Physics", "T002"),
        ("Chemistry", "T003"),
        ("English", "T004"),
        ("Biology", "T005"),
    ]

    days_of_week = ["Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"]

    student_data = []
    teacher_data = []

    for cls in classes:
        class_name = cls["class_name"]
        section = cls["section"]

        for day in days_of_week:
            if day in ["Saturday", "Sunday"]:
                # Fully empty day
                student_data.append({
                    "branch": branch,
                    "class_name": class_name,
                    "section": section,
                    "subject": "",
                    "teacher_employee_id": "",
                    "day_of_week": day,
                    "start_time": "",
                    "end_time": ""
                })
                teacher_data.append({
                    "branch": branch,
                    "teacher_employee_id": "",
                    "subject": "",
                    "class_name": class_name,
                    "section": section,
                    "day_of_week": day,
                    "start_time": "",
                    "end_time": ""
                })
            else:
                num_periods = random.randint(2, 4)  # number of periods per day
                start_hour = 9
                for i in range(num_periods):
                    subj, base_tid = random.choice(subjects)
                    teacher_id = f"{base_tid}_{section}"  # unique per section
                    start_time = f"{start_hour + i}:00"
                    end_time = f"{start_hour + i + 1}:00"

                    # Randomly make some periods empty
                    if random.random() < 0.2:
                        subj = ""
                        teacher_id = ""

                    # Student-side
                    student_data.append({
                        "branch": branch,
                        "class_name": class_name,
                        "section": section,
                        "subject": subj,
                        "teacher_employee_id": teacher_id,
                        "day_of_week": day,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    # Teacher-side
                    teacher_data.append({
                        "branch": branch,
                        "teacher_employee_id": teacher_id,
                        "subject": subj,
                        "class_name": class_name,
                        "section": section,
                        "day_of_week": day,
                        "start_time": start_time,
                        "end_time": end_time
                    })

    # ---------------- LEGEND SHEET ----------------
    legend_data = [
        {"Column": "branch", "Description": "Name of the branch/school"},
        {"Column": "class_name", "Description": "Class number or name"},
        {"Column": "section", "Description": "Section of the class"},
        {"Column": "subject",
            "Description": "Subject name (leave empty for empty period)"},
        {"Column": "teacher_employee_id",
            "Description": "Teacher's employee ID (leave empty for empty period)"},
        {"Column": "day_of_week",
            "Description": f"Day of the week. Must be one of {days_of_week}"},
        {"Column": "start_time", "Description": "Start time in HH:MM format"},
        {"Column": "end_time", "Description": "End time in HH:MM format"},
        {"Column": "", "Description": "Leave Saturday/Sunday rows empty to indicate fully empty day"}
    ]

    df_student = pd.DataFrame(student_data)
    df_teacher = pd.DataFrame(teacher_data)
    df_legend = pd.DataFrame(legend_data)

    # ---------------- WRITE TO EXCEL ----------------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_student.to_excel(writer, index=False,
                            sheet_name='Student_Timetable')
        df_teacher.to_excel(writer, index=False,
                            sheet_name='Teacher_Timetable')
        df_legend.to_excel(writer, index=False, sheet_name='Legend')
        writer.save()
    buffer.seek(0)

    response = HttpResponse(
        buffer,
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
        start_time = parse_time_str(data["start_time"]) if "start_time" in data else entry.start_time
        end_time = parse_time_str(data["end_time"]) if "end_time" in data else entry.end_time

        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        teacher = entry.teacher
        school_class = entry.school_class

        # --------- TEACHER CONFLICT CHECK ----------
        teacher_conflicts = Timetable.objects.filter(
            teacher=teacher,
            day_of_week=day
        ).exclude(id=entry.id)

        for t in teacher_conflicts:
            if time_overlap(start_time, end_time, t.start_time, t.end_time):
                raise ValueError("Teacher has overlapping period")

        # --------- CLASS CONFLICT CHECK ----------
        class_conflicts = Timetable.objects.filter(
            school_class=school_class,
            day_of_week=day
        ).exclude(id=entry.id)

        for c in class_conflicts:
            if time_overlap(start_time, end_time, c.start_time, c.end_time):
                raise ValueError("Class has overlapping period")

        # --------- ATOMIC UPDATE ----------
        with transaction.atomic():
            entry.subject = subject
            entry.day_of_week = day
            entry.start_time = start_time
            entry.end_time = end_time
            entry.save()

            # Re-sync Rust scheduler
            rust_backend.reschedule_class(
                class_id=f"{school_class.class_name}-{school_class.section}"
            )

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

            # Re-sync Rust scheduler
            rust_backend.reschedule_class(
                class_id=f"{school_class.class_name}-{school_class.section}"
            )

    except Exception:
        logger.exception("Timetable deletion failed")
        return JsonResponse({"error": "Failed to delete entry"}, status=500)

    return JsonResponse({"success": True})