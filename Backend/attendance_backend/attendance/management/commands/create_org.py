# attendance/management/commands/create_org.py

from django.core.management.base import BaseCommand
import hashlib
from attendance.models import Organization, Branch
from Admin.models import AdminAccessKey

class Command(BaseCommand):
    help = "Create organization with default admin key"

    def handle(self, *args, **kwargs):
        name = input("Organization name: ")
        org_type = input("Type (school/college/institute): ")
        branch_name = input("Branch name: ")

        org = Organization.objects.create(
            name=name,
            org_type=org_type
        )

        Branch.objects.create(
            organization=org,
            name=branch_name
        )

        default_combo = f"{name[:2]}{branch_name[:2]}↑".upper()
        combo_hash = hashlib.sha256(default_combo.encode()).hexdigest()

        AdminAccessKey.objects.create(
            organization=org,
            combo_hash=combo_hash,
            is_default=True
        )

        self.stdout.write(self.style.SUCCESS(
            f"Org created. Default admin key: {default_combo}"
        ))
