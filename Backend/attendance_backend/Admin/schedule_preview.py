# api/schedule_preview.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils.rust_client import _post_json, RustAPIError
from datetime import datetime, time

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def parse_time_str(s: str) -> time:
    return datetime.strptime(s, "%H:%M").time()

@api_view(["POST"])
def schedule_preview(request):
    """
    React/JS-friendly API: preview schedule using Rust scheduler.
    Expects JSON payload:
    {
        "classes": [
            {
                "class_name": "10",
                "section": "A",
                "subject": "Math",
                "day": "Monday",
                "start_time": "09:00",
                "end_time": "09:45"
            },
            ...
        ],
        "mode": "beam"
    }
    """
    data = request.data
    mode = data.get("mode", "normal")
    classes = []

    errors = []
    for c in data.get("classes", []):
        try:
            subject = c.get("subject") or "EMPTY"
            day = c.get("day")
            if day not in VALID_DAYS:
                raise ValueError(f"Invalid day: {day}")
            classes.append({
                "class_name": c["class_name"],
                "section": c["section"],
                "subject": subject,
                "start_time": c["start_time"],
                "end_time": c["end_time"]
            })
        except Exception as e:
            errors.append(f"Invalid entry {c}: {e}")

    if errors:
        return Response({"success": False, "errors": errors})

    # Call Rust scheduler
    try:
        resp = _post_json("/schedule/classes", {"classes": classes, "mode": mode})
    except RustAPIError as e:
        return Response({"success": False, "errors": [str(e)]})

    return Response({"success": True, "schedule": resp})