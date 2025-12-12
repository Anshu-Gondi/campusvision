from django.urls import path
from . import views, api

urlpatterns = [
    # ====================== STUDENT MANAGEMENT ======================
    path("students/", views.list_students, name="list_students"),
    path("students/register/", views.register_student, name="register_student"),
    path("students/delete/<str:roll_no>/", views.delete_student, name="delete_student"),
    path("students/upload/<int:student_id>/", views.upload_student_image, name="upload_student_image"),
    path("students/image/delete/<str:roll_no>/", views.delete_student_image, name="delete_student_image"),

    # ====================== TEACHER MANAGEMENT ======================
    path("teachers/", views.list_teachers, name="list_teachers"),
    path("teachers/register/", views.register_teacher, name="register_teacher"),
    path("teachers/delete/<str:employee_id>/", views.delete_teacher, name="delete_teacher"),
    path("teachers/upload/<int:teacher_id>/", views.upload_teacher_image, name="upload_teacher_image"),
    path("teachers/image/delete/<str:employee_id>/", views.delete_teacher_image, name="delete_teacher_image"),

    # ====================== RUST FACE RECOGNITION (NEW ERA) ======================
    path("face/register/", views.register_face_rust, name="register_face_rust"),
    path("face/verify/", views.verify_identity_rust, name="verify_identity_rust"),

    # ====================== CCTV CAMERA MANAGEMENT ======================
    path("cameras/", views.list_cameras, name="list_cameras"),
    path("cameras/add/", views.add_camera, name="add_camera"),
    path("cameras/delete/<int:camera_id>/", views.delete_camera, name="delete_camera"),
    path("cameras/live-cctv/", views.live_cctv, name="live_cctv"),

    # ====================== QR SYSTEM ======================
    path("qr/create/", views.create_qr_session, name="create_qr_session"),
    path("qr/validate/<uuid:code>/", views.validate_qr_session, name="validate_qr_session"),

    # ====================== ANALYTICS ======================
    path("analytics/students/", views.student_analytics, name="student_analytics"),
    path("analytics/teachers/", views.teacher_analytics, name="teacher_analytics"),
    path("analytics/student/<int:student_id>/", views.student_detail_analytics, name="student_detail_analytics"),
    path("analytics/teacher/<int:teacher_id>/", views.teacher_detail_analytics, name="teacher_detail_analytics"),

    # ====================== ADMIN AUTHENTICATION ======================
    path("api/admin/login/", api.admin_login),
]