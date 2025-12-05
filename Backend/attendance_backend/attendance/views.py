from datetime import timedelta
from queue import Queue
import threading
from django.http import StreamingHttpResponse
from django.conf import settings
from .models import CameraMatch, QRSession, Attendance, Student, StudentTimeTable, Teacher, TeacherTimeTable
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import get_object_or_404
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from django.shortcuts import get_object_or_404, redirect
from django.db.models import Count, Q
from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from django.utils.timezone import now
from .models import Student, Teacher, Attendance, QRSession, StudentTimeTable, TeacherTimeTable, Camera
from .serializers import StudentSerializer, StudentTimeTableSerializer, TeacherSerializer, AttendanceSerializer, QRSessionSerializer, TeacherTimeTableSerializer
from rust_backend import (
    detect_and_embed, add_person, search_person,
    get_face_info, save_database, total_registered,
    cctv_process_frame, cctv_clear_daily
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

# ----------------- Background Frame Queues (updated) -----------------
FRAME_QUEUES = {}      # camera_id -> Queue(maxsize=5)
WORKER_THREADS = {}    # camera_id -> Thread

# ----------------- Multi-Camera Attendance Logic (you already had this) -----------------
def evaluate_multi_camera(person_id):
    time_limit = now() - timedelta(seconds=10)
    matches = CameraMatch.objects.filter(
        person_id=person_id,
        timestamp__gte=time_limit
    ).values("camera").distinct().count()

    total_cameras = Camera.objects.filter(active=True).count()
    if total_cameras == 0:
        return False
    required = int((0.8 * total_cameras) + 0.999)  # 80% of active cameras
    return matches >= required


# ----------------- Camera Worker (BEST OF BOTH WORLDS) -----------------
def camera_worker(camera_id, url, role="student"):
    q = Queue(maxsize=5)
    FRAME_QUEUES[camera_id] = q
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}: {url}")
        return

    print(f"[INFO] Camera {camera_id} started → Role: {role.upper()} | URL: {url}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Camera {camera_id} lost connection. Reconnecting...")
            cap = cv2.VideoCapture(url)
            continue

        try:
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_bytes = buffer.tobytes()

            # Use our new RUST CCTV ENGINE (with tracking + once-per-day)
            results = cctv_process_frame(
                frame_bytes=img_bytes,
                role=role,
                min_confidence=0.68,
                min_track_hits=10
            )

            # Process all tracked + identified faces
            for person in results:
                bbox = person["bbox"]
                x, y, w, h = map(int, bbox)
                name = person.get("name", "Unknown")
                roll_no = person.get("roll_no", "")
                confidence = person.get("confidence", 0.0)
                mark_now = person.get("mark_now", False)
                identified = person.get("identified", False)

                # Draw box
                color = (0, 255, 0) if identified else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({confidence:.2f})"
                if mark_now:
                    label += " [MARKED]"
                    cv2.putText(frame, "ATTENDANCE MARKED!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)

                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # ----------------- YOUR ORIGINAL MULTI-CAMERA LOGIC -----------------
                if identified and roll_no:
                    CameraMatch.objects.create(
                        person_id=roll_no,
                        camera_id=camera_id,
                        similarity=confidence
                    )

                    # Mark attendance using your original 80% camera rule
                    if evaluate_multi_camera(roll_no):
                        if role == "student":
                            student = Student.objects.filter(roll_no=roll_no).first()
                            if student:
                                Attendance.objects.get_or_create(
                                    student=student,
                                    date=now().date(),
                                    defaults={"status": "Present", "time": now().time()}
                                )
                        else:
                            teacher = Teacher.objects.filter(employee_id=roll_no).first()
                            if teacher:
                                Attendance.objects.get_or_create(
                                    teacher=teacher,
                                    date=now().date(),
                                    defaults={"status": "Present", "time": now().time()}
                                )

        except Exception as e:
            print(f"[ERROR] CCTV processing failed on camera {camera_id}: {e}")

        # Send frame to frontend
        if q.full():
            q.get()
        q.put(frame)


# ----------------- Start Background Camera (now supports role) -----------------
def start_camera_thread(camera):
    if camera.id not in WORKER_THREADS:
        # Optional: Add `role` field to Camera model, fallback to "student"
        role = getattr(camera, 'role', 'student') if hasattr(camera, 'role') else "student"
        t = threading.Thread(
            target=camera_worker,
            args=(camera.id, camera.url, role),
            daemon=True
        )
        t.start()
        WORKER_THREADS[camera.id] = t
        print(f"[STARTED] Camera {camera.name} (ID: {camera.id}) → Role: {role.upper()}")


# ----------------- Live CCTV Stream (unchanged) -----------------
def generate_frames(camera_id):
    q = FRAME_QUEUES.get(camera_id)
    if not q:
        return
    while True:
        if q.empty():
            continue
        frame = q.get()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ----------------- Daily Reset at Midnight -----------------
def reset_daily_attendance():
    cctv_clear_daily()  # Clears Rust's internal daily tracker
    print(f"[RESET] Daily CCTV attendance cleared at {now()}")

def start_daily_reset():
    def run():
        while True:
            now_time = now()
            next_midnight = (now_time + timedelta(days=1)).replace(hour=0, minute=0, second=10, microsecond=0)
            sleep_time = (next_midnight - now_time).total_seconds()
            threading.Event().wait(sleep_time)
            reset_daily_attendance()
    threading.Thread(target=run, daemon=True).start()

# Start daily reset on server startup
start_daily_reset()

# ----------------- CCTV Camera APIs -----------------

@api_view(["POST"])
def add_camera(request):
    name = request.data.get("name")
    url = request.data.get("url")
    if not name or not url:
        return Response({"error": "Name and URL are required"}, status=400)
    cam = Camera.objects.create(name=name, url=url)
    return Response({"message": f"Camera {cam.name} added successfully", "id": cam.id}, status=201)

@api_view(["GET"])
def list_cameras(request):
    cameras = Camera.objects.filter(active=True)
    data = [{"id": cam.id, "name": cam.name, "url": cam.url} for cam in cameras]
    return Response(data)

@api_view(["DELETE"])
def delete_camera(request, camera_id):
    try:
        cam = Camera.objects.get(id=camera_id)
        cam.delete()
        return Response({"message": f"Camera {cam.name} deleted successfully"}, status=200)
    except Camera.DoesNotExist:
        return Response({"error": "Camera not found"}, status=404)

@api_view(["GET"])
def live_cctv(request):
    camera_id = int(request.GET.get("camera_id"))
    try:
        camera = Camera.objects.get(id=camera_id, active=True)
    except Camera.DoesNotExist:
        return Response({"error": "Camera not found"}, status=404)

    # Start background thread if not running
    start_camera_thread(camera)

    return StreamingHttpResponse(
        generate_frames(camera.id),
        content_type="multipart/x-mixed-replace;boundary=frame"
    )
