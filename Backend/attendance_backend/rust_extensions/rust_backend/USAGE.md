
# Rust Backend Extension – Usage Guide for Python Developers

This guide explains how to install and use the high-performance Rust backend (exposed via PyO3) in your Python projects.  
The extension provides fast face detection, recognition, tracking, enrollment, duplicate checking, database persistence, class scheduling, and utility functions for a real-time face recognition attendance system.

The Python module name is defined in your `Cargo.toml` under `[lib] name = "rust_backend"`.  
If you used a different name, replace `rust_backend` below with that name.

## Installation

### Prerequisites
- Python 3.8 or higher
- Rust toolchain (install via `rustup`)
- Maturin: `pip install maturin`

### Build and Install

```bash
# From the root of the Rust extension directory
maturin develop --release    # Fast development mode (recompiles quickly)
# OR build a wheel for distribution
maturin build --release
pip install target/wheels/*.whl
```

After installation:

```python
import rust_backend
```

## CCTV Real-Time Processing

### `cctv_process_frame(frame_bytes: bytes, role: str, min_confidence: float, min_track_hits: int) -> List[dict]`

Processes a single frame for face detection, recognition, and tracking.

**Parameters**  
- `frame_bytes`: JPEG-encoded image bytes (e.g., from OpenCV)
- `role`: `"student"` or `"teacher"`
- `min_confidence`: Minimum similarity for identification (0.0–1.0, typical 0.6)
- `min_track_hits`: Minimum consecutive hits before trusting a track (typical 5)

**Returns**  
List of dictionaries with keys:
```python
{
    "track_id": int,
    "bbox": [x1, y1, x2, y2],
    "hits": int,
    "age": int,
    "person_id": int | None,
    "name": str | None,
    "roll_no": str | None,
    "role": str,
    "identified": bool,
    "confidence": float,
    "mark_now": bool   # True when attendance should be recorded
}
```

**Example**
```python
import cv2
import rust_backend

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    success, buffer = cv2.imencode('.jpg', frame)
    results = rust_backend.cctv_process_frame(buffer.tobytes(), "student", 0.6, 5)
    for person in results:
        if person["mark_now"]:
            print(f"Attendance marked: {person['name']} ({person['roll_no']})")
```

### `cctv_get_tracked_faces(role: str) -> List[dict]`

Returns currently tracked faces.

**Returns**  
Similar structure to `cctv_process_frame` but without `"mark_now"`.

### `cctv_clear_daily() -> None`

Resets daily tracking state (call once per day, e.g., at midnight).

```python
rust_backend.cctv_clear_daily()
```

## Face Recognition & Enrollment

### `add_person(embedding: List[float], name: str, person_id: int, roll_no: str, role: str) -> int`

Adds a person and returns the internal index ID.

### `add_to_index(embedding: List[float], person_id: int, name: str, roll_no: str, role: str) -> int`

Equivalent to `add_person`, useful for bulk operations.

### `search_person(embedding: List[float], role: str, k: int = 5) -> List[Tuple[int, float]]`

Returns top-k matches: `[(person_id, similarity), ...]` (higher similarity = better).

### `can_reenroll(embedding: List[float], person_id: int, role: str) -> bool`

Checks if a new embedding is sufficiently different from existing ones for the same person (helps avoid low-quality duplicates).

### `query_similar(embedding: List[float], k: int = 5) -> List[int]`

Returns top-k similar person IDs across **all** roles (useful for global duplicate detection).

## Database Management

### `check_duplicate(embedding: List[float], role: str, threshold: float = 0.6) -> dict`

Checks for existing similar faces during registration.

**Returns**
```python
{
    "duplicate": bool,
    "matched_id": int | None,
    "similarity": float | None,
    "name": str | None,
    "roll_no": str | None
}
```

### `get_face_info(id: int) -> dict`

```python
{"id": int, "name": str, "roll_no": str, "role": str}
```
Raises `ValueError` if ID not found.

### Counters

```python
rust_backend.count_students() -> int
rust_backend.count_teachers() -> int
rust_backend.total_registered() -> int
```

### Persistence

```python
rust_backend.save_database("path/to/database.bin")  # No return
rust_backend.load_database("path/to/database.bin")  # No return
```

## Class Scheduling

Both functions accept the same input format and return assigned teachers.

### `schedule_classes(classes: List[dict]) -> List[dict]`

Greedy scheduling algorithm.

### `schedule_classes_beam(classes: List[dict]) -> List[dict]`

Beam search (usually better assignments, slightly slower).

**Input format**
```python
classes = [
    {
        "class_name": "Class 10A",
        "section": "A",
        "subject": "Mathematics",
        "start_time": "09:00",   # HH:MM format
        "end_time": "10:00"
    },
    # ...
]
```

**Output format**
```python
{
    "class_name": str,
    "section": str,
    "subject": str,
    "teacher_id": int,
    "teacher_name": str,
    "similarity": float,
    "reliability": float,
    "workload": float
}
```

**Example**
```python
assignments = rust_backend.schedule_classes_beam(classes)
for a in assignments:
    print(f"{a['subject']} → {a['teacher_name']} (sim: {a['similarity']:.2f})")
```

## Utilities

### `verify_face(input_image: bytes, known_embedding: List[float]) -> float`

1:1 verification similarity score (higher = more similar).

### `detect_emotion(input_image: bytes) -> int`

Returns emotion class ID from the face crop (exact mapping depends on your model).

## Best Practices

- Use JPEG-encoded bytes for `frame_bytes` and `input_image` for optimal speed.
- Call `save_database()` periodically in production.
- Run `cctv_clear_daily()` daily via cron or scheduled task.
- Wrap calls in try-except to handle `ValueError` or other PyO3 exceptions.
- All heavy computation runs in Rust — ideal for real-time CCTV pipelines.

## Example: Registration Flow

```python
# During new student registration
result = rust_backend.check_duplicate(embedding, "student", threshold=0.65)
if result["duplicate"]:
    print(f"Possible duplicate: {result['name']} ({result['roll_no']})")
else:
    if rust_backend.can_reenroll(embedding, person_id, "student"):
        rust_backend.add_person(embedding, name, person_id, roll_no, "student")
        rust_backend.save_database("faces.db")
        print("Successfully enrolled")
```

This Rust extension delivers blazing-fast, production-ready face recognition directly from Python.  
Happy coding!
