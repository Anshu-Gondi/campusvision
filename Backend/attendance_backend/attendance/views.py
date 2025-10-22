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
from deepface import DeepFace
from .models import Student, Teacher, Attendance, QRSession
from .serializers import StudentSerializer, StudentTimeTableSerializer, TeacherSerializer, AttendanceSerializer, QRSessionSerializer, TeacherTimeTableSerializer

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


# ----------------- Face Verification -----------------
# ----------------- Face Verification (DeepFace, no TensorFlow) -----------------


@api_view(["POST"])
def verify_identity(request):
    """
    Verify face + detect emotion for liveness.
    Secure QR: expiry + single-use + binding.
    """
    user_type = request.data.get("type")
    user_id = request.data.get("id")
    qr_code = request.data.get("qr_code")

    if not qr_code:
        return Response({"error": "QR code required"}, status=400)

    # ---------------- Validate QR Session ----------------
    try:
        session = QRSession.objects.get(code=qr_code)

        if not session.is_valid():
            return Response({"error": "QR session expired or already used"}, status=400)

        if session.used:
            return Response({"error": "QR session already used"}, status=400)

    except QRSession.DoesNotExist:
        return Response({"error": "QR session not found"}, status=404)

    if "image" not in request.FILES:
        return Response({"error": "No image file provided"}, status=400)

    # ---------------- Read Image ----------------
    file = request.FILES["image"]
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return Response({"error": "Invalid image file"}, status=400)

    # Convert BGR → RGB (DeepFace usually expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- Fetch Person ----------------
    if user_type == "student":
        person = get_object_or_404(Student, roll_no=user_id)
    elif user_type == "teacher":
        person = get_object_or_404(Teacher, employee_id=user_id)
    else:
        return Response({"error": "Invalid type"}, status=400)

    if not person.image:
        return Response({"error": "No image registered"}, status=400)

    # ---------------- Parallel Verification ----------------
    def face_match():
        return DeepFace.verify(
            img1_path=rgb_frame,
            img2_path=person.image.path,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )

    def emotion_check():
        return DeepFace.analyze(
            img_path=rgb_frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        face_future = executor.submit(face_match)
        emotion_future = executor.submit(emotion_check)

        try:
            face_result = face_future.result(timeout=5)
            emotion_result = emotion_future.result(timeout=5)
        except Exception as e:
            return Response({"error": f"Analysis failed: {str(e)}"}, status=500)

    # ---------------- Check Results ----------------
    if not face_result.get("verified", False):
        return Response({"error": "Face does not match"}, status=404)

    # Handle DeepFace emotion result safely (dict vs list)
    if isinstance(emotion_result, list):
        emotion = emotion_result[0].get("dominant_emotion", "unknown")
    else:
        emotion = emotion_result.get("dominant_emotion", "unknown")

    # Basic liveness → reject blank/unknown
    if emotion.lower() in ["unknown", "blank"]:
        return Response({"error": "Liveness failed: suspicious image"}, status=403)

    # ---------------- Mark Attendance ----------------
    if user_type == "student":
        Attendance.objects.get_or_create(
            student=person,
            date=now().date(),
            defaults={"status": "Present"}
        )
    elif user_type == "teacher":
        Attendance.objects.get_or_create(
            teacher=person,
            date=now().date(),
            defaults={"status": "Present"}
        )

    # End QR session (single-use)
    session.used = True
    session.scanned_by = user_id
    session.expires_at = now()
    session.save()

    return Response({
        "message": f"{user_type.capitalize()} {person.name} verified with liveness ✅",
        "emotion": emotion,
        "session_status": "ended"
    }, status=200)


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
