# CampusVision — Face Recognition Attendance System

CampusVision is a proprietary, production-ready face recognition attendance platform combining:
- a high-performance Rust engine (exposed to Python) for face preprocessing, embedding extraction, ANN searches and CCTV tracking,
- a Python/Django backend that orchestrates business logic and persistence,
- a Vite-based web frontend for admin and organization workflows (schools, colleges, institutions),
- an Expo / React Native mobile app for end-users (students, teachers, employees).

**Important**: This repository and its contents are the intellectual property of the owner (Anshu-Gondi). The code and binaries are proprietary. Any modification, redistribution, or reuse requires explicit permission from the owner — see the LICENSE file at the repository root.

### Table of Contents
- [Project Overview](#project-overview)
- [Repository Layout (Summary)](#repository-layout-summary)
- [Component Roles](#component-roles)
- [rust_backend (High-Level Technical Summary)](#rust_backend-the-rust-engine-concise-technical-summary)
- [Quick Start Pointers (Local/Dev)](#quick-local-run-pointers-summary)
- [Security & Privacy](#security--privacy)
- [Repository Ownership, Change Policy, and Contact](#repository-ownership-change-policy-and-contact)
- [License](#license)

---

## Project Overview
CampusVision automates attendance by recognizing faces from image uploads, manual captures, or live CCTV streams. The system is designed for educational institutions and organizations, delivering low-latency inference, robust multi-frame tracking, and fast nearest-neighbor matching using an HNSW index. The Rust extension provides the performance-critical core and is seamlessly integrated into the Django backend.

---

## Repository Layout (Summary)
```
Face-Recognition-Attendance-System/
├── Backend/
│   ├── attendance_backend/              # Main Django project (settings, urls, wsgi, etc.)
│   │   ├── attendance_backend/          # Project folder (inner)
│   │   ├── attendance/                  # Django app: attendance logic, models, views
│   │   ├── Admin/                       # Django app: organization management, admin features
│   │   ├── rust_extensions/
│   │   │   └── rust_backend/            # Rust crate (pyo3/maturin) exposing performance primitives
│   │   ├── manage.py                    # Django management script
│   │   ├── requirements.txt             # Python dependencies
│   │   └── start scripts / env files    # Development scripts and configuration
├── frontend/                                # Vite + React web UI – Admin & organization dashboards/workflows
├── mobile/                                  # Expo / React Native (TypeScript) – User-facing mobile application
├── LICENSE                                  # Project license (proprietary)
└── README.md                                # This file
```

---

## Component Roles
- **Backend (Django)**: Handles API endpoints, business logic, database persistence, authentication, and integration with the Rust engine.
  - `attendance_backend/attendance_backend/` – Django project configuration (settings, urls, etc.).
  - `attendance_backend/attendance/` – App focused on attendance marking, records, and reporting.
  - `attendance_backend/Admin/` – App handling organization management, user roles, class/section setup, and admin workflows.
  - `attendance_backend/rust_extensions/rust_backend/` – High-performance Rust core.
- **frontend/**: Web dashboard built with Vite + React, primarily for administrators, teachers, and institution staff to manage students, classes, reports, CCTV feeds, and system settings.
- **mobile/**: React Native (Expo) mobile application for end-users (students/teachers) – features include profile viewing, attendance history, notifications, and quick check-ins.

---

## rust_backend — The Rust Engine (Concise Technical Summary)
**Location**: `Backend/attendance_backend/rust_extensions/rust_backend`

**Purpose**: Implements performance-sensitive operations and exposes them to Python via pyo3 (built with maturin).

**Key Responsibilities**
- Image preprocessing and face alignment (OpenCV + YuNet).
- Model inference (TorchScript via tch-rs preferred, ONNX fallback).
- Fast ANN search with HNSW (separate indices for students and teachers).
- CCTV frame tracking, IoU + embedding matching, daily attendance deduplication.
- Persistence of embeddings and metadata (serde + bincode).
- Scheduling algorithms exposed to Python.
- Comprehensive Python API via `src/py_functions/*`.

**Python API Examples**
- `detect_and_embed`, `detect_and_add_person`
- `add_person`, `search_person`, `check_duplicate`
- `cctv_process_frame`, `cctv_get_tracked_faces`
- `save_database`, `load_database`
- `schedule_classes`, `verify_face`, `detect_emotion`

**Build & Install**
```bash
cd Backend/attendance_backend/rust_extensions/rust_backend
pip install maturin
maturin develop --release    # Installs into active venv
```

---

## Quick Local Run Pointers (Summary)

**Backend**
- Create and activate a Python virtual environment.
- `pip install -r Backend/attendance_backend/requirements.txt`
- Build and install `rust_backend` (see above).
- Run migrations and start Django server:
  ```bash
  cd Backend/attendance_backend
  python manage.py migrate
  python manage.py runserver
  ```

**Frontend (Admin/Organization Dashboard)**
```bash
cd frontend
pnpm install    # or npm/yarn install
pnpm dev        # Starts Vite dev server
```

**Mobile (User App)**
```bash
cd mobile
pnpm install
pnpm start      # Expo start – scan QR or use emulator
```

---

## Security & Privacy
- Face images and embeddings are sensitive biometric data. Always:
  - Obtain explicit consent before collection.
  - Use HTTPS everywhere.
  - Encrypt data at rest and in transit.
  - Restrict access and maintain audit logs.
- Never commit datasets, model weights, or secrets to version control.

---

## Repository Ownership, Change Policy, and Contact
- **Ownership**: All content is proprietary and owned by Anshu-Gondi.
- **Change Policy**: Modifications, forks, or contributions require explicit written permission. Open an issue for any requests.
- **Contact**:
  - GitHub: [https://github.com/Anshu-Gondi](https://github.com/Anshu-Gondi)
  - For permission or collaboration inquiries, open an issue or use the owner's preferred contact method.

---

## License
See the [LICENSE](LICENSE) file. The project is proprietary with "All Rights Reserved." Unauthorized copying, modification, distribution, or use is prohibited.

---

**Made with ❤️ by [Anshu Gondi](https://github.com/Anshu-Gondi)**
