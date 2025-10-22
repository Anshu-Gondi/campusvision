from rest_framework import serializers
from .models import Student, StudentTimeTable, Teacher, Attendance, QRSession, TeacherTimeTable

# ----------------- Student Serializer -----------------


class StudentSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()  # full URL for frontend

    class Meta:
        model = Student
        fields = "__all__"  # includes original image field and image_url

    def get_image_url(self, obj):
        request = self.context.get("request")
        if obj.image and request:
            return request.build_absolute_uri(obj.image.url)
        elif obj.image:
            return obj.image.url  # fallback
        return None

    def validate_image(self, value):
        if value and value.size > 2 * 1024 * 1024:  # 2MB limit
            raise serializers.ValidationError("Image size must be under 2MB.")
        return value


# ----------------- Teacher Serializer -----------------
class TeacherSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()  # full URL for frontend

    class Meta:
        model = Teacher
        fields = "__all__"

    def get_image_url(self, obj):
        request = self.context.get("request")
        if obj.image and request:
            return request.build_absolute_uri(obj.image.url)
        elif obj.image:
            return obj.image.url
        return None

    def validate_image(self, value):
        if value and value.size > 2 * 1024 * 1024:
            raise serializers.ValidationError("Image size must be under 2MB.")
        return value


# ----------------- Attendance Serializer -----------------
class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = "__all__"

# ----------------- Student Timetable Serializer -----------------


class StudentTimeTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentTimeTable
        fields = "__all__"


# ----------------- Teacher Timetable Serializer -----------------
class TeacherTimeTableSerializer(serializers.ModelSerializer):
    teacher_name = serializers.CharField(source="teacher.name", read_only=True)

    class Meta:
        model = TeacherTimeTable
        fields = "__all__"

# ----------------- QR Code Serializer -----------------


class QRSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = QRSession
        fields = ["code", "created_at", "expires_at"]
