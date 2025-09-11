import os
import uuid
from django.db import models
from datetime import datetime
from django.utils.timezone import now, timedelta
from django.core.exceptions import ValidationError
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver

# ----------------- Student -----------------
class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_no = models.CharField(max_length=20, unique=True)
    class_name = models.CharField(max_length=50)
    section = models.CharField(max_length=10)
    image = models.ImageField(upload_to="students/", null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.roll_no})"


# ----------------- Teacher -----------------
class Teacher(models.Model):
    name = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=20, unique=True)
    subject = models.CharField(max_length=100)
    class_name = models.CharField(max_length=50)
    section = models.CharField(max_length=10)
    image = models.ImageField(upload_to="teachers/", null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.employee_id})"


# ----------------- Attendance -----------------
class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, null=True, blank=True)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE, null=True, blank=True)
    date = models.DateField(default=datetime.now)
    time = models.TimeField(default=datetime.now)
    status = models.CharField(max_length=10, choices=[("Present", "Present"), ("Absent", "Absent")])

    def clean(self):
        if not self.student and not self.teacher:
            raise ValidationError("Attendance must be linked to either a student or a teacher.")

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

    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = now() + timedelta(minutes=5)
        super().save(*args, **kwargs)

    def is_valid(self):
        return now() < self.expires_at

    def __str__(self):
        return f"{self.code} valid till {self.expires_at}"

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