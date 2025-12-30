from django.urls import path
from . import views

urlpatterns = [
    # ====================== ADMIN AUTHENTICATION ======================
    path("login/", views.admin_login, name="admin-login"),

    # ORGANIZATION
    path("organization/", views.get_organization),
    path("organization/update/", views.update_organization),

    # SECURITY
    path("organization/rotate-combo/", views.rotate_admin_combo),

    # ================= BRANCHES =================
    path("branches/", views.branches_view),
    path("branches/<int:branch_id>/", views.branch_detail_view),

    # ================= TEACHERS =================
    path("teachers/", views.teachers_view),
    path("teachers/<int:teacher_id>/", views.teacher_detail_view),
    path("teachers/<int:teacher_id>/image/", views.admin_upload_teacher_image),
    path("teachers/<int:teacher_id>/image/delete/", views.admin_delete_teacher_image),

    # ================ STUDENTS =================
    path("students/", views.students_view),
    path("students/<int:student_id>/", views.student_detail_view),
    path("students/<int:student_id>/image/", views.admin_upload_student_image),
    path("students/<int:student_id>/image/delete/", views.admin_delete_student_image),
]
