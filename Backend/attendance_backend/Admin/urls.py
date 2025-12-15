from django.urls import path
from . import views

urlpatterns = [
    # ====================== ADMIN AUTHENTICATION ======================
    path("login/", views.admin_login, name="admin-login"),
]