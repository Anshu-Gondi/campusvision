import os
import uuid
from django.db import models
from datetime import datetime
from django.utils.timezone import now, timedelta
from django.core.exceptions import ValidationError
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver

# ----------------- Organization (School / College) -----------------

class Organization(models.Model):
    ORG_TYPES = [
        ("school", "School"),
        ("college", "College"),
        ("institute", "Institute"),
    ]

    name = models.CharField(max_length=200)
    org_type = models.CharField(max_length=20, choices=ORG_TYPES)
    website = models.URLField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.org_type})"


# ----------------- Branch / Campus -----------------

class Branch(models.Model):
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)   # e.g., "Main Campus", "Delhi Branch", "Campus A"

    address = models.CharField(max_length=500, null=True, blank=True)
    city = models.CharField(max_length=100, null=True, blank=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    pincode = models.CharField(max_length=20, null=True, blank=True)

    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.organization.name} - {self.name}"


# ----------------- Department (optional but useful) -----------------

class Department(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.branch.name} - {self.name}"


# ----------------- Student -----------------


class Student(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=100)
    roll_no = models.CharField(max_length=20, unique=True)
    class_name = models.CharField(max_length=50)
    section = models.CharField(max_length=10)
    image = models.ImageField(upload_to="students/", null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.roll_no})"


# ----------------- Teacher -----------------
class Teacher(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=20, unique=True)

    # A teacher can teach multiple subjects
    subjects = models.JSONField(default=list)  # ["Math", "Physics"]

    # Can handle multiple class levels
    can_teach_classes = models.JSONField(default=list)  # ["10-A", "10-B", "9-A"]

    # How reliable the teacher is (based on historical attendance)
    reliability_score = models.FloatField(default=0.8)

    # Used to avoid overloading the same teacher
    workload_score = models.IntegerField(default=0)

    # Optional vector embedding for compatibility search
    embedding_vector = models.JSONField(default=list, blank=True)

    image = models.ImageField(upload_to="teachers/", null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.employee_id})"

class SchoolClass(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    class_name = models.CharField(max_length=50, null=True, blank=True)   # "10"
    section = models.CharField(max_length=10, null=True, blank=True)      # "A"

    def __str__(self):
        return f"{self.class_name}-{self.section}"

# ----------------- Student Timetable -----------------


class StudentTimeTable(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    school_class = models.ForeignKey(SchoolClass, on_delete=models.CASCADE, null=True, blank=True)
    subject = models.CharField(max_length=100)
    day_of_week = models.CharField(
        max_length=10,
        choices=[("Monday", "Monday"), ("Tuesday", "Tuesday"), ("Wednesday", "Wednesday"),
                 ("Thursday", "Thursday"), ("Friday", "Friday"), ("Saturday", "Saturday"), ("Sunday", "Sunday")]
    )
    start_time = models.TimeField()
    end_time = models.TimeField()

    def __str__(self):
        return f"{self.school_class} | {self.subject} | {self.day_of_week} {self.start_time}-{self.end_time}"


# ----------------- Teacher Timetable -----------------

class TeacherTimeTable(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100)
    school_class = models.ForeignKey(SchoolClass, on_delete=models.CASCADE, null=True, blank=True)
    day_of_week = models.CharField(
        max_length=10,
        choices=[("Monday", "Monday"), ("Tuesday", "Tuesday"), ("Wednesday", "Wednesday"),
                 ("Thursday", "Thursday"), ("Friday", "Friday"), ("Saturday", "Saturday"), ("Sunday", "Sunday")]
    )
    start_time = models.TimeField()
    end_time = models.TimeField()

    def __str__(self):
        return f"{self.teacher.name} | {self.school_class} | {self.subject} | {self.day_of_week} {self.start_time}-{self.end_time}"

# ----------------- Attendance -----------------


class Attendance(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, null=True, blank=True)
    teacher = models.ForeignKey(
        Teacher, on_delete=models.CASCADE, null=True, blank=True)
    student_timetable = models.ForeignKey(
        StudentTimeTable, on_delete=models.SET_NULL, null=True, blank=True)
    teacher_timetable = models.ForeignKey(
        TeacherTimeTable, on_delete=models.SET_NULL, null=True, blank=True)
    date = models.DateField(default=datetime.now)
    time = models.TimeField(default=datetime.now)
    status = models.CharField(max_length=10, choices=[
                              ("Present", "Present"), ("Absent", "Absent")])

    def clean(self):
        if not self.student and not self.teacher:
            raise ValidationError(
                "Attendance must be linked to either a student or a teacher.")

    def __str__(self):
        if self.student:
            return f"{self.student.name} - {self.date} - {self.status}"
        elif self.teacher:
            return f"{self.teacher.name} - {self.date} - {self.status}"
        return f"Attendance {self.date}"


# ----------------- QR CODE SESSION -----------------
class QRSession(models.Model):
    code = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    grace_seconds = 5  # allow overlap scanning
    used = models.BooleanField(default=False)  # <-- NEW
    # student roll_no or teacher id
    scanned_by = models.CharField(max_length=100, null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = now() + timedelta(seconds=20)
        super().save(*args, **kwargs)

    def is_valid(self):
        if self.used:
            return False
        return now() < (self.expires_at + timedelta(seconds=self.grace_seconds))

    def __str__(self):
        return f"{self.code} valid till {self.expires_at}"

# -----------------  CCTV Camera -----------------
class Camera(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    url = models.CharField(max_length=500)
    floor = models.CharField(max_length=50, null=True, blank=True)
    room = models.CharField(max_length=50, null=True, blank=True)
    active = models.BooleanField(default=True)
    role = models.CharField(max_length=10, choices=[("student", "Student"), ("teacher", "Teacher")], default="student")

class CameraMatch(models.Model):
    person_id = models.CharField(max_length=100)  # roll_no or employee_id
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    similarity = models.FloatField()
    

# -----------------  CCTV Camera -----------------

class AdminAccessKey(models.Model):
    combo_hash = models.CharField(max_length=256)
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional: limit to only 1 record
    def save(self, *args, **kwargs):
        if not AdminAccessKey.objects.exists():
            super().save(*args, **kwargs)
        else:
            raise Exception("Only one admin key allowed")


# Delete old image file on update
@receiver(pre_save, sender=Student)
def auto_delete_old_student_image(sender, instance, **kwargs):
    if not instance.pk:
        return  # new object, nothing to delete

    try:
        old_image = Student.objects.get(pk=instance.pk).image
    except Student.DoesNotExist:
        return

    new_image = instance.image
    if old_image and old_image != new_image:
        if os.path.isfile(old_image.path):
            os.remove(old_image.path)


@receiver(pre_save, sender=Teacher)
def auto_delete_old_teacher_image(sender, instance, **kwargs):
    if not instance.pk:
        return

    try:
        old_image = Teacher.objects.get(pk=instance.pk).image
    except Teacher.DoesNotExist:
        return

    new_image = instance.image
    if old_image and old_image != new_image:
        if os.path.isfile(old_image.path):
            os.remove(old_image.path)


# Delete image file on delete
@receiver(post_delete, sender=Student)
def delete_student_image(sender, instance, **kwargs):
    if instance.image and os.path.isfile(instance.image.path):
        os.remove(instance.image.path)


@receiver(post_delete, sender=Teacher)
def delete_teacher_image(sender, instance, **kwargs):
    if instance.image and os.path.isfile(instance.image.path):
        os.remove(instance.image.path)
