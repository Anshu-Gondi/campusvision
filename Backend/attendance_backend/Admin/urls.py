from django.urls import path
from . import views, timetable_views, schedule_views, user_upload_views

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
    path("teachers/<int:teacher_id>/image/delete/",
         views.admin_delete_teacher_image),

    # ================ STUDENTS =================
    path("students/", views.students_view),
    path("students/<int:student_id>/", views.student_detail_view),
    path("students/<int:student_id>/image/", views.admin_upload_student_image),
    path("students/<int:student_id>/image/delete/",
         views.admin_delete_student_image),


    # ================= USER UPLOADS (TEACHER & STUDENT DATA) =================
    path("user-upload/teachers/", user_upload_views.upload_teachers_excel),
    path("user-upload/students/", user_upload_views.upload_students_excel),
    path("user-upload/download-sample-teachers/",
         user_upload_views.download_sample_teachers_excel),
    path("user-upload/download-sample-students/",
         user_upload_views.download_sample_students_excel),

    # ================= TIMETABLE =================
    path("timetable/", timetable_views.list_timetable),
    path("timetable/upload/", timetable_views.upload_timetable_smart),
    path("timetable/download-sample-time-table",
         timetable_views.download_sample_timetable),
    path("timetable/<int:timetable_id>/update/",
         timetable_views.update_timetable_entry),
    path("timetable/<int:timetable_id>/delete/",
         timetable_views.delete_timetable_entry),

    # ================= SCHEDULE & SUBSTITUTIONS =================
    path("scheduler/substitution/", schedule_views.generate_substitution),
]
