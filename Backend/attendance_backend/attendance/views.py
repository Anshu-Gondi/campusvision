from django.http import StreamingHttpResponse
from django.conf import settings
from .models import QRSession, Attendance, Student, StudentTimeTable, Teacher, TeacherTimeTable
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import get_object_or_404
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
import face_recognition
from django.shortcuts import get_object_or_404, redirect
from django.db.models import Count, Q
from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from django.utils.timezone import now
from .models import Student, Teacher, Attendance, QRSession
from .serializers import StudentSerializer, StudentTimeTableSerializer, TeacherSerializer, AttendanceSerializer, QRSessionSerializer, TeacherTimeTableSerializer
from rust_backend import (
    detect_and_embed, add_person, search_person,
    get_face_info, save_database, total_registered
)

# ----------------- CRUD ViewSets -----------------


class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer


class TeacherViewSet(viewsets.ModelViewSet):
    queryset = Teacher.objects.all()
    serializer_class = TeacherSerializer


class AttendanceViewSet(viewsets.ModelViewSet):
    queryset = Attendance.objects.all()
    serializer_class = AttendanceSerializer


# ----------------- Timetable APIs -----------------
@api_view(["GET"])
def student_timetable(request, class_name, section):
    timetable = StudentTimeTable.objects.filter(
        class_name=class_name, section=section)
    serializer = StudentTimeTableSerializer(timetable, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def teacher_timetable(request, teacher_id):
    timetable = TeacherTimeTable.objects.filter(
        teacher__employee_id=teacher_id)
    serializer = TeacherTimeTableSerializer(timetable, many=True)
    return Response(serializer.data)


# ----------------- Student APIs -----------------
@api_view(["GET"])
def list_students(request):
    students = Student.objects.all()
    serializer = StudentSerializer(
        students, many=True, context={"request": request})
    return Response(serializer.data)


@api_view(["POST"])
def register_student(request):
    serializer = StudentSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------- Upload / Update Student Image -----------------


@api_view(["POST"])
def upload_student_image(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    serializer = StudentSerializer(
        student, data=request.data, partial=True)  # allow partial update
    if serializer.is_valid():
        serializer.save()
        return Response({"message": f"Image updated for student {student.name}"}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------- Delete Student Image -----------------


@api_view(["DELETE"])
def delete_student_image(request, roll_no):
    try:
        student = Student.objects.get(roll_no=roll_no)
        if student.image:
            student.image.delete(save=False)  # Django deletes the file too
            student.image = None
            student.save()
            return Response({"message": "Student image deleted successfully"}, status=status.HTTP_200_OK)
        return Response({"error": "No image found for this student"}, status=status.HTTP_404_NOT_FOUND)
    except Student.DoesNotExist:
        return Response({"error": "Student not found"}, status=status.HTTP_404_NOT_FOUND)

# ----------------- Delete Student Record (with attendance + image cleanup) -----------------


@api_view(["DELETE"])
def delete_student(request, roll_no):
    try:
        student = Student.objects.get(roll_no=roll_no)

        # Delete related attendance records
        Attendance.objects.filter(student=student).delete()

        # Delete image file if exists
        if student.image:
            student.image.delete(save=False)

        student.delete()
        return Response({"message": f"Student {roll_no} and related attendance deleted successfully"}, status=status.HTTP_200_OK)

    except Student.DoesNotExist:
        return Response({"error": "Student not found"}, status=status.HTTP_404_NOT_FOUND)

# ----------------- Teacher APIs -----------------


@api_view(["GET"])
def list_teachers(request):
    teachers = Teacher.objects.all()
    serializer = TeacherSerializer(
        teachers, many=True, context={"request": request})
    return Response(serializer.data)


@api_view(["POST"])
def register_teacher(request):
    serializer = TeacherSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------- Upload / Update Teacher Image -----------------


@api_view(["POST"])
def upload_teacher_image(request, teacher_id):
    teacher = get_object_or_404(Teacher, id=teacher_id)
    serializer = TeacherSerializer(teacher, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": f"Image updated for teacher {teacher.name}"}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------- Delete Teacher Image -----------------


@api_view(["DELETE"])
def delete_teacher_image(request, employee_id):
    try:
        teacher = Teacher.objects.get(employee_id=employee_id)
        if teacher.image:
            teacher.image.delete(save=False)
            teacher.image = None
            teacher.save()
            return Response({"message": "Teacher image deleted successfully"}, status=status.HTTP_200_OK)
        return Response({"error": "No image found for this teacher"}, status=status.HTTP_404_NOT_FOUND)
    except Teacher.DoesNotExist:
        return Response({"error": "Teacher not found"}, status=status.HTTP_404_NOT_FOUND)

# ----------------- Delete Teacher Record (with attendance + image cleanup) -----------------


@api_view(["DELETE"])
def delete_teacher(request, employee_id):
    try:
        teacher = Teacher.objects.get(employee_id=employee_id)

        # Delete related attendance records
        Attendance.objects.filter(teacher=teacher).delete()

        # Delete image file if exists
        if teacher.image:
            teacher.image.delete(save=False)

        teacher.delete()
        return Response({"message": f"Teacher {employee_id} and related attendance deleted successfully"}, status=status.HTTP_200_OK)

    except Teacher.DoesNotExist:
        return Response({"error": "Teacher not found"}, status=status.HTTP_404_NOT_FOUND)


# ----------------- QR CODE SESSION -----------------


@api_view(["POST"])
def create_qr_session(request):
    """
    Create a new QR session for attendance.
    """
    session = QRSession.objects.create()
    serializer = QRSessionSerializer(session)
    return Response(serializer.data, status=201)


@api_view(["GET"])
def validate_qr_session(request, code):
    """
    Validate QR session code and optionally redirect to React frontend.
    """
    try:
        session = QRSession.objects.get(code=code)

        if not session.is_valid():
            return Response({"valid": False, "error": "Expired or already used"}, status=400)

        # Mark as used immediately
        session.used = True
        # optional (student roll_no/teacher_id)
        session.scanned_by = request.GET.get("user")
        session.save()

        # Handle redirect (React frontend)
        redirect_url = request.GET.get("redirect")
        if redirect_url:
            return redirect(f"{redirect_url}?code={code}")

        return Response({"valid": True, "code": str(session.code)})

    except QRSession.DoesNotExist:
        return Response({"valid": False, "error": "Not found"}, status=404)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def checkin_attendance(request):
    session_id = request.data.get("sessionId")
    student_id = request.user.username   # assuming username = roll_no

    try:
        session = QRSession.objects.get(code=session_id)
        if not session.is_valid():
            return Response({"error": "QR expired or already used"}, status=400)

        # Prevent reuse
        session.used = True
        session.scanned_by = student_id
        session.save()

        # Mark attendance
        student = get_object_or_404(Student, roll_no=student_id)
        Attendance.objects.create(
            student=student, status="Present", date=now().date())

        return Response({"message": "Attendance marked successfully"})
    except QRSession.DoesNotExist:
        return Response({"error": "Invalid session"}, status=404)


# ----------------- Analytics -----------------
# ----------------- Student Analytics -----------------


@api_view(["GET"])
def student_analytics(request):
    students = Student.objects.annotate(
        total=Count("attendance"),
        present=Count("attendance", filter=Q(attendance__status="Present"))
    )

    student_data = [
        {
            "student": s.name,
            "roll_no": s.roll_no,
            "attendance_percent": (s.present / s.total * 100) if s.total else 0
        }
        for s in students
    ]

    most_absent_student = min(
        student_data, key=lambda x: x["attendance_percent"], default=None)

    return Response({
        "attendance_data": student_data,
        "most_absent": most_absent_student
    })


# ----------------- Teacher Analytics -----------------
@api_view(["GET"])
def teacher_analytics(request):
    teachers = Teacher.objects.annotate(
        total=Count("attendance"),
        present=Count("attendance", filter=Q(attendance__status="Present"))
    )

    teacher_data = [
        {
            "teacher": t.name,
            "employee_id": t.employee_id,
            "attendance_percent": (t.present / t.total * 100) if t.total else 0
        }
        for t in teachers
    ]

    most_absent_teacher = min(
        teacher_data, key=lambda x: x["attendance_percent"], default=None)

    return Response({
        "attendance_data": teacher_data,
        "most_absent": most_absent_teacher
    })


# ----------------- Single Student Analytics -----------------
@api_view(["GET"])
def student_detail_analytics(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    total = Attendance.objects.filter(student=student).count()
    present = Attendance.objects.filter(
        student=student, status="Present").count()
    percent = (present / total * 100) if total > 0 else 0

    return Response({
        "student": student.name,
        "roll_no": student.roll_no,
        "attendance_percent": percent,
        "total_classes": total,
        "present_count": present,
        "absent_count": total - present,
    })


# ----------------- Single Teacher Analytics -----------------
@api_view(["GET"])
def teacher_detail_analytics(request, teacher_id):
    teacher = get_object_or_404(Teacher, id=teacher_id)
    total = Attendance.objects.filter(teacher=teacher).count()
    present = Attendance.objects.filter(
        teacher=teacher, status="Present").count()
    percent = (present / total * 100) if total > 0 else 0

    return Response({
        "teacher": teacher.name,
        "employee_id": teacher.employee_id,
        "attendance_percent": percent,
        "total_classes": total,
        "present_count": present,
        "absent_count": total - present,
    })


@api_view(["POST"])
def register_face_rust(request):
    """Register student/teacher using Rust HNSW"""
    user_type = request.data.get("type")  # "student" or "teacher"
    user_id = request.data.get("id")      # roll_no or employee_id
    name = request.data.get("name")

    if user_type not in ["student", "teacher"]:
        return Response({"error": "type must be student/teacher"}, status=400)

    if "image" not in request.FILES:
        return Response({"error": "Image required"}, status=400)

    file = request.FILES["image"]
    img_bytes = file.read()

    result = detect_and_embed(img_bytes)
    if not result["found"]:
        return Response({"error": "No face detected"}, status=400)

    embedding = result["embedding"]
    face_id = add_person(embedding, name, user_id, user_type)

    # optional: save every time or every 10 mins
    save_database(str(settings.FACE_DATABASE_PATH))

    return Response({
        "message": f"{user_type.capitalize()} registered",
        "face_id": face_id,
        "total_registered": total_registered()
    })


@api_view(["POST"])
def verify_identity_rust(request):
    """Lightning-fast verification using Rust HNSW"""
    user_type = request.data.get("type", "student")
    qr_code = request.data.get("qr_code")

    if "image" not in request.FILES:
        return Response({"error": "Image required"}, status=400)

    # Optional: validate QR
    if qr_code:
        try:
            session = QRSession.objects.get(code=qr_code)
            if not session.is_valid() or session.used:
                return Response({"error": "Invalid/used QR"}, status=400)
            session.used = True
            session.save()
        except QRSession.DoesNotExist:
            return Response({"error": "Invalid QR"}, status=404)

    file = request.FILES["image"]
    result = detect_and_embed(file.read())

    if not result["found"]:
        return Response({"error": "No face detected"}, status=400)

    embedding = result["embedding"]
    matches = search_person(embedding, user_type, k=1)

    if not matches or matches[0][1] < 0.6:  # similarity threshold
        return Response({"error": "Face not recognized"}, status=404)

    face_id = matches[0][0]
    info = get_face_info(face_id)

    # Mark attendance
    if user_type == "student":
        student = get_object_or_404(Student, roll_no=info["roll_no"])
        Attendance.objects.get_or_create(
            student=student, date=now().date(), defaults={"status": "Present"})
    else:
        teacher = get_object_or_404(Teacher, employee_id=info["roll_no"])
        Attendance.objects.get_or_create(
            teacher=teacher, date=now().date(), defaults={"status": "Present"})

    return Response({
        "message": "Verified!",
        "name": info["name"],
        "role": info["role"],
        "similarity": round(matches[0][1], 3)
    })


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()

        result = detect_and_embed(img_bytes)
        if result["found"]:
            x, y, w, h = result["bbox"]
            embedding = result["embedding"]

            for role in ["student", "teacher"]:
                matches = search_person(embedding, role, k=1)
                if matches and matches[0][1] > 0.6:
                    info = get_face_info(matches[0][0])
                    label = f"{info['name']} ({role})"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@api_view(["GET"])
def live_cctv(request):
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace;boundary=frame")
