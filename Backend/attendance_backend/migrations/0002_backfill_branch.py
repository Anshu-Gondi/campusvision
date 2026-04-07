from django.db import migrations


def backfill_branch(apps, schema_editor):
    FaceImage = apps.get_model('attendance', 'FaceImage')
    CameraMatch = apps.get_model('attendance', 'CameraMatch')
    FaceRejectionLog = apps.get_model('attendance', 'FaceRejectionLog')
    QRSession = apps.get_model('attendance', 'QRSession')
    Student = apps.get_model('attendance', 'Student')
    Teacher = apps.get_model('attendance', 'Teacher')
    Branch = apps.get_model('attendance', 'Branch')

    default_branch = Branch.objects.first()
    if not default_branch:
        return  # no branches → nothing to fix

    # 🔥 Preload maps (CRITICAL)
    student_map = {
        s.roll_no: s.branch_id
        for s in Student.objects.all().only("roll_no", "branch_id")
    }

    teacher_map = {
        t.employee_id: t.branch_id
        for t in Teacher.objects.all().only("employee_id", "branch_id")
    }

    # ---------- FaceImage ----------
    updates = []
    for img in FaceImage.objects.filter(branch__isnull=True):
        branch_id = None

        if img.person_type == 'student':
            branch_id = student_map.get(img.person_id)
        elif img.person_type == 'teacher':
            branch_id = teacher_map.get(img.person_id)

        img.branch_id = branch_id or default_branch.id
        updates.append(img)

    FaceImage.objects.bulk_update(updates, ["branch"])

    # ---------- CameraMatch ----------
    updates = []
    for cam in CameraMatch.objects.filter(branch__isnull=True):
        branch_id = student_map.get(cam.person_id) or teacher_map.get(cam.person_id)
        cam.branch_id = branch_id or default_branch.id
        updates.append(cam)

    CameraMatch.objects.bulk_update(updates, ["branch"])

    # ---------- FaceRejectionLog ----------
    updates = []
    for log in FaceRejectionLog.objects.filter(branch__isnull=True):
        branch_id = student_map.get(log.person_id) or teacher_map.get(log.person_id)
        log.branch_id = branch_id or default_branch.id
        updates.append(log)

    FaceRejectionLog.objects.bulk_update(updates, ["branch"])

    # ---------- QRSession ----------
    updates = []
    for qr in QRSession.objects.filter(branch__isnull=True):
        branch_id = student_map.get(qr.scanned_by) or teacher_map.get(qr.scanned_by)
        qr.branch_id = branch_id or default_branch.id
        updates.append(qr)

    QRSession.objects.bulk_update(updates, ["branch"])


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0031_remove_camera_floor_remove_camera_room_and_more'),
    ]

    operations = [
        migrations.RunPython(backfill_branch),
    ]