from django.db import models
from attendance.models import Organization

class AdminAccessKey(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="admin_keys"
    )

    combo_hash = models.CharField(max_length=256)
    active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"AdminKey for {self.organization.name}"
