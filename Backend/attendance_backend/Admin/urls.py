from django.urls import path
from . import views

urlpatterns = [
    # ====================== ADMIN AUTHENTICATION ======================
    path("admin/login/", views.admin_login),
]