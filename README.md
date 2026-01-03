# CampusVision — Face Recognition Attendance System

CampusVision is a proprietary, production-ready face recognition attendance platform combining:
- a high-performance Rust engine (exposed to Python) for face preprocessing, embedding extraction, ANN searches and CCTV tracking,
- a Python/Django backend that orchestrates business logic and persistence,
- a Vite-based web frontend, and
- an Expo / React Native mobile app.

Important: This repository and its contents are the intellectual property of the owner (Anshu‑Gondi). The code and binaries are proprietary. Any modification, redistribution, or reuse requires explicit permission from the owner — see the LICENSE file at the repository root.

Table of contents
- Project overview
- Repository layout (summary)
- rust_backend (high-level technical summary)
- Quick start pointers (local/dev)
- Security & privacy
- How to request permission / contact
- License

---

## Project overview
CampusVision automates attendance by recognizing faces from uploads or live CCTV streams. The project focuses on low-latency inference, robust tracking across frames, and fast nearest-neighbour matching using an HNSW index. The Rust extension crate provides the performance-critical functionality and is consumed by the Python backend (Django).

---

## Repository layout (top-level)
- Backend/
  - attendance_backend/ — Django (or Python) backend and application code.
  - rust_extensions/rust_backend/ — Rust crate (pyo3/maturin) that exposes the performance primitives to Python.
  - requirements.txt, start scripts, and environment files expected here.
- frontend/ — Vite web UI (source in `src/`, public assets in `public/`).
- mobile/ — Expo / React Native app (TypeScript).
- LICENSE — project license (proprietary / "All rights reserved" summary included in rust_backend README).

---

## rust_backend — the Rust engine (concise technical summary)
Location: Backend/attendance_backend/rust_extensions/rust_backend

Purpose: the Rust crate implements performance-sensitive operations and exposes them to Python via pyo3 (packaged with maturin). It is intended to be built and installed into the Python environment used by the Django backend.

Key responsibilities
- Image preprocessing and alignment (OpenCV + YuNet); returns model-ready tensors.
- Model inference orchestration (prefers TorchScript via tch, falls back to ONNX runtime).
- Fast approximate nearest neighbour (ANN) search using HNSW (hnsw_rs) with separate indices for students and teachers.
- CCTV tracking primitives: per-camera trackers, IoU + embedding similarity matching, daily mark avoidance, and track lifecycle management.
- Persistence helpers to save / load embeddings and metadata (serde + bincode).
- Scheduling algorithms (teacher-class assignment) implemented in Rust and exposed to Python.
- Python-facing API implemented in src/py_functions/* (face_recognition, detection, cctv, database, scheduler, utils).

Important modules (source-level)
- src/preprocess.rs — decode, detect (YuNet ONNX), align, produce tch::Tensor ([1,3,160,160]).
- src/hnsw_helper.rs — manages two HNSW indices, metadata, embeddings; add/search/save/load functions.
- src/cctv_tracker.rs — tracking logic, embedding/emotion extraction, matching logic.
- src/cctv_state.rs — simple in-memory "already-marked-today" registry.
- src/py_functions/* — pyo3 wrappers exposing the Rust functions to Python as the `rust_backend` module.

Python API (examples, exported names)
- detect_and_embed / detect_and_add_person
- add_person, search_person, can_reenroll, add_to_index
- cctv_process_frame, cctv_get_tracked_faces, cctv_clear_daily
- check_duplicate, get_face_info, count_students/teachers, save_database, load_database
- schedule_classes, schedule_classes_beam
- verify_face, detect_emotion

How Django integrates (high level)
- Build the extension (see Build below) and install in the Python environment used by Django.
- Import `rust_backend` in Django code and call exposed functions from views, background workers, or Celery tasks:
  - Enrollment: use detect_and_add_person or detect_and_embed + add_person.
  - Live CCTV processing: call cctv_process_frame and persist identified results as Attendance records.
  - Duplicate-check during enrollment: check_duplicate.

Build & install (developer/dev env)
- Prerequisites: Rust (stable), Python 3.10+, maturin, platform native libraries (OpenCV, libtorch if needed), and a Python virtualenv.
- Common development commands (run inside a Python venv):
  - cd Backend/attendance_backend/rust_extensions/rust_backend
  - pip install maturin
  - maturin develop --release        # builds and installs the extension into current venv
  - (alternatively) maturin build --release && pip install target/wheels/*.whl
- Notes:
  - Ensure the system has the necessary native libraries if building with native OpenCV / libtorch support.
  - Use the same Python interpreter for maturin and Django (ABI compatibility).

Quick local run pointers (summary)
- Backend:
  - Create and activate a Python virtual environment.
  - pip install -r Backend/requirements.txt
  - Build and install rust_backend as above (if required by backend).
  - Run Django dev server / start application per Backend README.
- Frontend:
  - cd frontend
  - pnpm install   (or npm/yarn)
  - pnpm dev       (or npm run dev)
- Mobile:
  - cd mobile
  - pnpm install
  - pnpm start     (expo start)
  - Use Expo App / emulator to run.

---

## Security & privacy (short)
- Face images and embeddings are sensitive biometric data. You must:
  - Obtain explicit consent before collecting face data.
  - Use HTTPS for all transport channels.
  - Encrypt data at rest and in transit.
  - Properly restrict access and audit exports.
- Do not commit face datasets, model weights, or secrets to the repository.

---

## Repository ownership, change policy, and contact
- Ownership: repository content and rust_backend are proprietary and owned by Anshu‑Gondi.
- Change policy: any changes to the repository, code, or license require explicit permission from the owner. If you need updates, obtain written permission from Anshu‑Gondi or open a request/issue in this repository for review/approval.
- Contact / permission requests:
  - GitHub: https://github.com/Anshu-Gondi
  - For permission requests, open an issue tagging the owner or reach out via the owner’s preferred contact.

---

## License
- See the LICENSE file in the repository root. The rust_backend README and license text indicate "All Rights Reserved" and state that copying, modifying, or redistributing the code is prohibited without explicit authorization.

---

If you want this README adjusted (tone, length, or to include/omit specific technical details), tell me the exact changes and I will produce a revised version. I will not commit anything unless you explicitly request and authorize me to do so.
