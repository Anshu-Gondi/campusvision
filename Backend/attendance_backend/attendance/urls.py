from django.urls import path
from . import views

urlpatterns = [
    path("students/", views.list_students, name="list_students"),
    path("teachers/", views.list_teachers, name="list_teachers"),
    path("register/student/", views.register_student, name="register_student"),
    path("register/teacher/", views.register_teacher, name="register_teacher"),
    path("verify/", views.verify_identity, name="verify_identity"),

    # QR Code Create/ Session 
    path("qr/create/", views.create_qr_session, name="create_qr_session"),
    path("qr/validate/<uuid:code>/", views.validate_qr_session, name="validate_qr_session"),

    # Upload / Update Images
    path("upload/student/<int:student_id>/", views.upload_student_image, name="upload_student_image"),
    path("upload/teacher/<int:teacher_id>/", views.upload_teacher_image, name="upload_teacher_image"),

    # Image management
    path("delete/student/image/<str:roll_no>/", views.delete_student_image, name="delete_student_image"),
    path("delete/teacher/image/<str:employee_id>/", views.delete_teacher_image, name="delete_teacher_image"),
    # Analytics
    path("analytics/students/", views.student_analytics, name="student_analytics"),
    path("analytics/teachers/", views.teacher_analytics, name="teacher_analytics"),
    path("analytics/student/<int:student_id>/", views.student_detail_analytics, name="student_detail_analytics"),
    path("analytics/teacher/<int:teacher_id>/", views.teacher_detail_analytics, name="teacher_detail_analytics"),
]
