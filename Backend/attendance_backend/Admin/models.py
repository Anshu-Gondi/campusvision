from django.db import models

# Create your models here.
class AdminAccessKey(models.Model):
    combo_hash = models.CharField(max_length=256)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional: limit to only 1 record
    def save(self, *args, **kwargs):
        if not AdminAccessKey.objects.exists():
            super().save(*args, **kwargs)
        else:
            raise Exception("Only one admin key allowed")
