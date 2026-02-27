# user_upload_views.py (production-grade optimized)

import io
import json
import logging
from io import BytesIO
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, JsonResponse
from django.db import transaction
from attendance.models import Branch, Teacher, Student, Department
from .face_security import admin_required
from django.conf import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = [".xls", ".xlsx"]


def validate_file_extension(file_name: str):
    if not any(file_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError("Invalid file format")


@csrf_exempt
@admin_required
@require_http_methods(["POST"])
def upload_teachers_excel(request):
    org_id = request.admin["org_id"]
    preview_mode = request.GET.get("preview", "true").lower() == "true"

    if "file" not in request.FILES:
        return JsonResponse({"error": "Excel file required"}, status=400)

    file = request.FILES["file"]

    try:
        validate_file_extension(file.name)
        df = pd.read_excel(BytesIO(file.read()))
    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception:
        logger.exception("Failed to read Excel")
        return JsonResponse({"error": "Failed to read Excel file"}, status=400)

    required_columns = ["employee_id", "name", "branch"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        return JsonResponse({"error": f"Missing columns: {missing_cols}"}, status=400)

    results = []
    teachers_to_create = []

    branches_cache = {
        b.name.strip(): b
        for b in Branch.objects.filter(organization_id=org_id)
    }

    departments_cache = {
        d.name.strip(): d
        for d in Department.objects.filter(branch__organization_id=org_id)
    }

    # preload existing employee IDs per branch
    existing_teachers = set(
        Teacher.objects.filter(branch__organization_id=org_id)
        .values_list("branch_id", "employee_id")
    )

    for idx, row in df.iterrows():
        row_result = {"row": idx + 2}

        try:
            branch_name = str(row["branch"]).strip()
            branch = branches_cache.get(branch_name)
            if not branch:
                raise ValueError(f"Branch '{branch_name}' not found")

            employee_id = str(row["employee_id"]).strip()

            if (branch.id, employee_id) in existing_teachers:
                row_result["status"] = "skip"
                results.append(row_result)
                continue

            # Department resolve
            dept_name = str(row.get("department", "")).strip()
            department = departments_cache.get(dept_name) if dept_name else None

            # Parse JSON fields safely
            def parse_json_field(value):
                if not value:
                    return []
                if isinstance(value, list):
                    return value
                try:
                    return json.loads(value)
                except Exception:
                    return []

            subjects = parse_json_field(row.get("subjects"))
            can_teach_classes = parse_json_field(row.get("can_teach_classes"))

            teacher = Teacher(
                branch=branch,
                employee_id=employee_id,
                name=str(row["name"]).strip(),
                department=department,
                subjects=subjects,
                can_teach_classes=can_teach_classes,
            )

            teachers_to_create.append(teacher)
            row_result["status"] = "ok"

        except Exception as e:
            row_result["status"] = "fail"
            row_result["error"] = str(e)
            logger.exception(f"Row {idx + 2} teacher upload failed")

        results.append(row_result)

    if preview_mode:
        return JsonResponse({
            "preview": True,
            "results": results,
            "success_count": sum(r["status"] == "ok" for r in results)
        })

    try:
        with transaction.atomic():
            Teacher.objects.bulk_create(teachers_to_create)
    except Exception:
        logger.exception("Failed to commit teachers")
        return JsonResponse({"error": "Failed to save teachers"}, status=500)

    return JsonResponse({
        "preview": False,
        "results": results,
        "success_count": sum(r["status"] == "ok" for r in results)
    })

@csrf_exempt
@admin_required
@require_http_methods(["POST"])
def upload_students_excel(request):
    org_id = request.admin["org_id"]
    preview_mode = request.GET.get("preview", "true").lower() == "true"

    if "file" not in request.FILES:
        return JsonResponse({"error": "Excel file required"}, status=400)

    file = request.FILES["file"]

    try:
        validate_file_extension(file.name)
        df = pd.read_excel(BytesIO(file.read()))
    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception:
        logger.exception("Failed to read Excel")
        return JsonResponse({"error": "Failed to read Excel file"}, status=400)

    required_columns = ["roll_no", "name", "class_name", "section", "branch"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        return JsonResponse({"error": f"Missing columns: {missing_cols}"}, status=400)

    results = []
    students_to_create = []

    branches_cache = {
        b.name.strip(): b
        for b in Branch.objects.filter(organization_id=org_id)
    }

    departments_cache = {
        d.name.strip(): d
        for d in Department.objects.filter(branch__organization_id=org_id)
    }

    existing_students = set(
        Student.objects.filter(branch__organization_id=org_id)
        .values_list("branch_id", "roll_no")
    )

    for idx, row in df.iterrows():
        row_result = {"row": idx + 2}

        try:
            branch_name = str(row["branch"]).strip()
            branch = branches_cache.get(branch_name)
            if not branch:
                raise ValueError(f"Branch '{branch_name}' not found")

            roll_no = str(row["roll_no"]).strip()

            if (branch.id, roll_no) in existing_students:
                row_result["status"] = "skip"
                results.append(row_result)
                continue

            class_name = str(row["class_name"]).strip()
            section = str(row["section"]).strip()

            dept_name = str(row.get("department", "")).strip()
            department = departments_cache.get(dept_name) if dept_name else None

            student = Student(
                branch=branch,
                roll_no=roll_no,
                name=str(row["name"]).strip(),
                class_name=class_name,
                section=section,
                department=department,
            )

            students_to_create.append(student)
            row_result["status"] = "ok"

        except Exception as e:
            row_result["status"] = "fail"
            row_result["error"] = str(e)
            logger.exception(f"Row {idx + 2} student upload failed")

        results.append(row_result)

    if preview_mode:
        return JsonResponse({
            "preview": True,
            "results": results,
            "success_count": sum(r["status"] == "ok" for r in results)
        })

    try:
        with transaction.atomic():
            Student.objects.bulk_create(students_to_create)
    except Exception:
        logger.exception("Failed to commit students")
        return JsonResponse({"error": "Failed to save students"}, status=500)

    return JsonResponse({
        "preview": False,
        "results": results,
        "success_count": sum(r["status"] == "ok" for r in results)
    })

@csrf_exempt
@admin_required
@require_http_methods(["GET"])
def download_sample_teachers_excel(request):
    """
    Returns a sample Excel for bulk teacher upload:
    - Sheet1: Teachers
    - Sheet2: Legend explaining each column
    """
    # ---------------- SAMPLE BRANCHES ----------------
    branches = Branch.objects.filter(organization_id=request.admin["org_id"])
    branch_names = [b.name for b in branches] or ["Main Campus"]

    teacher_data = []
    for i in range(1, 6):
        teacher_data.append({
            "employee_id": f"T{i:03d}",
            "name": f"Teacher {i}",
            "branch": branch_names[i % len(branch_names)],
            "department": f"Department {i%3 + 1}",
            "subjects": json.dumps(["Mathematics", "Physics"] if i % 2 == 0 else ["English"]),
            "can_teach_classes": json.dumps(["10A", "10B"] if i % 2 == 0 else ["11A"])
        })

    legend_data = [
        {"Column": "employee_id", "Description": "Unique employee ID of the teacher"},
        {"Column": "name", "Description": "Full name of the teacher"},
        {"Column": "branch", "Description": "Branch name. Must match an existing branch"},
        {"Column": "department", "Description": "Department ID or name (optional)"},
        {"Column": "subjects", "Description": "JSON array of subjects the teacher can teach"},
        {"Column": "can_teach_classes", "Description": "JSON array of classes (like 10A, 10B) teacher can handle"}
    ]

    df_teachers = pd.DataFrame(teacher_data)
    df_legend = pd.DataFrame(legend_data)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_teachers.to_excel(writer, index=False, sheet_name="Teachers")
        df_legend.to_excel(writer, index=False, sheet_name="Legend")
    buffer.seek(0)

    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=sample_teachers.xlsx'
    return response


@csrf_exempt
@admin_required
@require_http_methods(["GET"])
def download_sample_students_excel(request):
    """
    Returns a sample Excel for bulk student upload:
    - Sheet1: Students
    - Sheet2: Legend explaining each column
    """
    branches = Branch.objects.filter(organization_id=request.admin["org_id"])
    branch_names = [b.name for b in branches] or ["Main Campus"]

    student_data = []
    classes = ["10", "11"]
    sections = ["A", "B"]

    roll_counter = 1
    for branch in branch_names:
        for cls in classes:
            for sec in sections:
                for i in range(1, 6):
                    student_data.append({
                        "roll_no": f"{roll_counter:03d}",
                        "name": f"Student {roll_counter}",
                        "class_name": cls,
                        "section": sec,
                        "branch": branch,
                        "department": f"Department {i%3 + 1}"
                    })
                    roll_counter += 1

    legend_data = [
        {"Column": "roll_no", "Description": "Unique roll number of the student"},
        {"Column": "name", "Description": "Full name of the student"},
        {"Column": "class_name", "Description": "Class/grade of the student"},
        {"Column": "section", "Description": "Section of the class"},
        {"Column": "branch", "Description": "Branch name. Must match an existing branch"},
        {"Column": "department", "Description": "Department ID or name (optional)"}
    ]

    df_students = pd.DataFrame(student_data)
    df_legend = pd.DataFrame(legend_data)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_students.to_excel(writer, index=False, sheet_name="Students")
        df_legend.to_excel(writer, index=False, sheet_name="Legend")
    buffer.seek(0)

    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=sample_students.xlsx'
    return response