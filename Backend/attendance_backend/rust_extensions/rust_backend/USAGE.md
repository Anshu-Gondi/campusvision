# Rust Backend Extension – Usage Guide for Python Developers

This guide explains how to install and use the high-performance Rust backend (exposed via PyO3) in your Python projects.  
The extension provides fast face detection, recognition, tracking, enrollment, duplicate checking, database persistence, class scheduling, and utility functions for a real-time face recognition attendance system.

> **Module name**: The imported Python module name is defined in `Cargo.toml` under `[lib] name = "..."`.  
> In this guide we use `rust_backend`. Replace it with your actual module name if different.

## Installation

### Prerequisites
- Python 3.8 or higher
- Rust toolchain (install via https://rustup.rs)
- Maturin: `pip install maturin`

### Build and Install

```bash
# Navigate to the Rust extension directory
cd path/to/rust_backend

# Development mode (fast reloads)
maturin develop --release

# OR build a distributable wheel
maturin build --release
pip install target/wheels/*.whl
```

After installation:

```python
import rust_backend
```

## CCTV Real-Time Processing

### `cctv_process_frame(frame_bytes: bytes, role: str, min_confidence: float, min_track_hits: int) -> List[dict]`

Processes a single video frame for detection, recognition, and tracking.

**Parameters**
- `frame_bytes`: JPEG-encoded image bytes (e.g., from OpenCV)
- `role`: `"student"` or `"teacher"`
- `min_confidence`: Minimum similarity threshold for identification (typical: 0.6)
- `min_track_hits`: Minimum consecutive detections before trusting a track (typical: 5)

**Returns**  
List of dictionaries, each containing:

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
    if success:
        results = rust_backend.cctv_process_frame(buffer.tobytes(), "student", 0.6, 5)
        for person in results:
            if person["mark_now"]:
                print(f"Attendance marked: {person['name']} ({person['roll_no']})")
```

### `cctv_get_tracked_faces(role: str) -> List[dict]`

Returns currently active tracked faces (same structure as above, without `"mark_now"`).

### `cctv_clear_daily() -> None`

Clears daily tracking state. Call once per day (e.g., at midnight).

```python
rust_backend.cctv_clear_daily()
```

## Face Recognition & Enrollment

### `add_person(embedding: List[float], name: str, person_id: int, roll_no: str, role: str) -> int`

Adds a person to the index and returns the internal index ID.

### `add_to_index(embedding: List[float], person_id: int, name: str, roll_no: str, role: str) -> int`

Same as `add_person` — useful for bulk indexing.

### `search_person(embedding: List[float], role: str, k: int = 5) -> List[Tuple[int, float]]`

Searches within the specified role. Returns top-k matches as `(person_id, similarity)` (higher similarity = better match).

### `can_reenroll(embedding: List[float], person_id: int, role: str) -> bool`

Checks whether the new embedding is sufficiently different from existing ones for the same person (prevents low-quality duplicates during re-enrollment).

### `query_similar(embedding: List[float], k: int = 5) -> List[int]`

Returns top-k similar person IDs across **all** roles (global duplicate detection).

## Database Management

### `check_duplicate(embedding: List[float], role: str, threshold: float = 0.6) -> dict`

Checks for duplicates during registration.

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
rust_backend.save_database("path/to/database.bin")
rust_backend.load_database("path/to/database.bin")
```

## Class Scheduling

Both functions take the same input and return teacher assignments.

### `schedule_classes(classes: List[dict]) -> List[dict]`

Greedy algorithm.

### `schedule_classes_beam(classes: List[dict]) -> List[dict]`

Beam search (generally better quality, slightly slower).

**Input format**

```python
classes = [
    {
        "class_name": "Class 10A",
        "section": "A",
        "subject": "Mathematics",
        "start_time": "09:00",   # HH:MM
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
    print(f"{a['subject']} → {a['teacher_name']} (similarity: {a['similarity']:.2f})")
```

## Utilities

### `verify_face(input_image: bytes, known_embedding: List[float]) -> float`

1:1 face verification — returns similarity score (higher = more similar).

### `detect_emotion(input_image: bytes) -> int`

Detects emotion from a face crop and returns the class ID (mapping depends on your trained model).

## Best Practices

- Always use JPEG-encoded bytes for `frame_bytes` and `input_image`.
- Persist the database regularly with `save_database()` in production.
- Schedule `cctv_clear_daily()` daily (cron, Celery beat, etc.).
- Wrap function calls in `try-except` to handle PyO3 exceptions (`ValueError`, etc.).
- All heavy operations run in Rust — perfect for real-time CCTV streams.

## Example: Full Registration Flow

```python
# During registration of a new student
result = rust_backend.check_duplicate(embedding, "student", threshold=0.65)

if result["duplicate"]:
    print(f"Possible duplicate: {result['name']} ({result['roll_no']})")
else:
    if rust_backend.can_reenroll(embedding, person_id, "student"):
        rust_backend.add_person(embedding, name, person_id, roll_no, "student")
        rust_backend.save_database("faces.db")
        print("Successfully enrolled!")
    else:
        print("Embedding too similar to existing ones — try another photo")
```

This Rust extension delivers **blazing-fast**, production-ready face recognition capabilities directly from Python.

Happy coding! 🚀


