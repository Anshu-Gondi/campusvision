<div align="center">

<img src="frontend/public/Leonardo_Phoenix_10_Minimalist_futuristic_logo_for_Attendance_3.jpg" alt="CampusVision Logo" width="120" height="120" style="border-radius: 16px;" />

# CampusVision

**AI-Powered Face Recognition Attendance Platform**

*Built for educational institutions that demand accuracy, speed, and scale.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-REST-092E20?style=flat-square&logo=django&logoColor=white)](https://djangoproject.com)
[![Rust](https://img.shields.io/badge/Rust-Vision%20Engine-CE422B?style=flat-square&logo=rust&logoColor=white)](https://rust-lang.org)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Expo](https://img.shields.io/badge/Expo-React%20Native-000020?style=flat-square&logo=expo&logoColor=white)](https://expo.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)](LICENSE)

</div>

---

## What is CampusVision?

CampusVision is a production-grade attendance management platform that replaces manual roll-calls with real-time face recognition. Whether through live CCTV feeds, continuously rotating QR codes, or manual image uploads — attendance is marked automatically, accurately, and with multiple fraud verification layers.

Designed for schools, colleges, and corporate campuses, it ships with three integrated interfaces: an **admin web dashboard**, a **teacher/student mobile app**, and a high-performance **Rust-based vision engine** that runs face detection and recognition at scale.

---

## Key Features

- 🎯 **Three-Layer Fraud Verification** — Face recognition → emotion liveness check (Emotion-8-FerPlus) → CCTV cross-verification. All three must pass before attendance is marked.
- 📡 **Live CCTV Mode** — Real-time CCTV stream processing with multi-face tracking across camera feeds; also serves as the final physical presence check in the verification pipeline
- 📱 **Rolling QR Code Attendance** — Continuously rotating short-lived QR codes (30–40 seconds per code) for teacher-managed sessions. Expired codes are immediately invalidated server-side — screenshot sharing and delayed proxy scans are structurally impossible
- 🏫 **Multi-Organization & Multi-Branch** — Full tenant isolation; manage multiple campuses from a single instance
- 📊 **Analytics Dashboard** — Attendance trends, class-wise reports, and exportable data
- 🔐 **Role-Based Access Control** — Admins, teachers, and students each see only what they need
- 🗂️ **Timetable-Aware Scheduling** — Attendance is validated against a live class timetable
- 🌍 **Geo-Fenced Check-ins** — Location validation to prevent proxy attendance
- 🔒 **Biometric Security** — Encrypted face embeddings stored in MinIO object storage with backup support
- 📦 **Docker Ready** — Full containerized deployment with `docker-compose`

---

## Attendance Verification Pipeline

CampusVision uses a three-stage pipeline before any attendance record is written. Every stage must pass — failure at any point is logged for manual review and attendance is not marked.

```
Student present in classroom
         │
         ▼
┌─────────────────────────┐
│  Stage 1: Face Recognition  │
│  YuNet detection            │
│  ArcFace embeddings         │
│  Rust HNSW index lookup     │
└────────────┬────────────┘
             │ Identity confirmed
             ▼
┌─────────────────────────────────┐
│  Stage 2: Liveness Check        │
│  Emotion-8-FerPlus model runs   │
│  Static photo/screen = flat     │
│  affect → rejected              │
│  Real person = natural affect   │
│  variation → passes             │
└────────────┬────────────────────┘
             │ Liveness confirmed
             ▼
┌──────────────────────────────────────┐
│  Stage 3: CCTV Cross-Verification    │
│  Recognized identity cross-checked   │
│  against CCTV feed                   │
│  Confirms physical presence in room  │
│  Catches remote/proxy attempts that  │
│  passed stages 1 and 2               │
└────────────┬─────────────────────────┘
             │ Presence confirmed
             ▼
      ✅ Attendance marked
```

**Why three stages?** Face recognition alone is one lock. A printed photo defeats it. The emotion model catches spoofed media — a static image or screen replay produces flat, unchanging affect that the model flags. CCTV verification catches the remaining attack: someone physically present in a different location who passed the first two stages remotely. Schools already have CCTV infrastructure — this adds verification without adding hardware cost.

---

## Rolling QR Code — How It Works

The QR attendance mode is built around the assumption that students will attempt to share codes. Static or long-lived QR codes are trivially defeated by a screenshot. CampusVision solves this with server-driven rotation:

1. **Teacher opens a session** — the backend generates the first QR token, scoped to the session ID, class context, and a server timestamp.
2. **Code rotates every 30–40 seconds** — the teacher's screen (web or mobile) automatically fetches and displays the current live token. The previous token is immediately invalidated server-side the moment the new one is issued.
3. **Student scans the active code** — the mobile app submits the scanned token to the backend. The server checks: is this the current valid token for this session? If it has expired or does not match, it is rejected outright — no grace period.
4. **One scan per student per session** — duplicate submissions are blocked at the backend regardless of token validity.
5. **Session closes** — the teacher ends the session; all remaining tokens are invalidated and the attendance record is locked.

A screenshot shared to an absent student is useless within 40 seconds. There is no window to exploit.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CampusVision                        │
├─────────────────┬──────────────────┬────────────────────┤
│  React Frontend │  Expo Mobile App │  Rust Vision Engine│
│  (Admin Panel)  │  (Teacher/Student│  (CCTV + Face AI)  │
│  Vite + React   │   React Native)  │  vision_engine     │
└────────┬────────┴────────┬─────────┴──────────┬─────────┘
         │                 │                     │
         ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│              Django REST API Backend                    │
│   attendance/ | Admin/ | org_apis | schedules           │
├─────────────────────────────────────────────────────────┤
│         PostgreSQL / SQLite    +    MinIO (Faces)       │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend API** | Python 3.10+, Django, DRF | REST API, business logic, ORM |
| **Vision Engine** | Rust (`vision_engine` crate) | Real-time face detection & recognition |
| **Face Models** | ONNX — YuNet, ArcFace, FaceNet | Detection & embedding generation |
| **Liveness Model** | ONNX — Emotion-8-FerPlus | Liveness detection via affect analysis |
| **Embedding Store** | Custom Rust HNSW index (`intelligence_core`) | Fast approximate nearest-neighbor search |
| **Object Storage** | MinIO | Encrypted face image & embedding storage |
| **Web Frontend** | React 18, Vite, CSS Modules | Admin dashboard |
| **Mobile App** | React Native, Expo, TypeScript | Teacher & student interface |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Relational data |
| **Deployment** | Docker, docker-compose | Containerized production setup |

---

## Project Structure

```
campus_vision_project/
│
├── Backend/
│   └── attendance_backend/
│       ├── attendance/              # Core attendance: models, APIs, serializers, geo utils
│       ├── Admin/                   # Org management, bulk upload, timetable, face security
│       ├── attendance_backend/      # Django config: settings, urls, wsgi/asgi
│       ├── campus_vision_engine/    # Rust workspace
│       │   ├── vision_engine/       #   HTTP server + camera pipeline + face recognition
│       │   ├── intelligence_core/   #   HNSW index, cosine similarity, SIMD math
│       │   └── intelligence_py/     #   PyO3 Python bindings for scheduling
│       ├── manage.py
│       └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Admin/               # Dashboard, Branches, Orgs, Timetable, Scheduler
│       │   ├── Attendance/          # Scanner, Camera, Form
│       │   ├── Analytics/           # Reports & trends
│       │   ├── Students/            # Student management
│       │   └── Teachers/            # Teacher management
│       └── services/api.js          # Centralized API client
│
├── mobile/
│   ├── app/                         # Expo Router screens & navigation
│   ├── components/                  # Shared UI components
│   ├── hooks/                       # Custom React Native hooks
│   └── constants/theme.ts           # Design tokens
│
├── docker/
│   └── docker-compose.yml           # Full-stack container orchestration
│
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ & pnpm
- Rust toolchain (`rustup`)
- Docker & Docker Compose (for production)

---

### 1 — Backend (Django API)

```bash
cd Backend/attendance_backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r ../requirements.txt

# Set up environment variables
cp .env.example .env            # Fill in your values

# Run database migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

> API available at `http://localhost:8000`

---

### 2 — Vision Engine (Rust)

```bash
cd Backend/attendance_backend/campus_vision_engine/vision_engine

# Copy environment config
cp .env.example .env

# Build and run (release mode for performance)
cargo run --release
```

> Vision engine binds on its own port (configured via `.env`). The Django backend communicates with it via the `rust_client.py` utility.

**Required ONNX models** (place in `vision_engine/models/`):

- `face_detection_yunet_2023mar.onnx`
- `arcface.onnx`
- `facenet.onnx`
- `emotion-ferplus-8.onnx`

---

### 3 — Frontend (Admin Dashboard)

```bash
cd frontend

# Install dependencies
pnpm install

# Copy and configure environment
cp .env.example .env

# Start dev server
pnpm dev
```

> Web dashboard available at `http://localhost:5173`

---

### 4 — Mobile App (Teacher & Student)

```bash
cd mobile

# Install dependencies
pnpm install

# Start Expo dev server
pnpm start
```

> Scan the QR code with **Expo Go** on your device, or run on an Android/iOS simulator.

---

### 5 — Full Stack with Docker

```bash
cd docker
cp .env.example .env   # Configure secrets

docker-compose up --build
```

This brings up Django, the Rust vision engine, MinIO, and any configured database in one command.

---

## Module Deep-Dive

### `attendance/` — Core Attendance Logic

Handles everything related to recording, querying, and managing attendance records.

- **Models:** `Student`, `Teacher`, `Attendance`, `QRSession`, `Camera`, `Branch`, `Organization`, `SchoolClass`, `Timetable`, `FaceImage`, `FaceRejectionLog`
- **Key APIs:** Mark attendance, generate and rotate QR sessions (30–40s token window), fetch reports, manage cameras
- **Utils:** Geo-fencing (`geo.py`), MinIO integration (`minio.py`), face database backup (`face_db_backup.py`), crypto helpers

### `Admin/` — Organization & Admin Workflows

- Bulk upload students/teachers with face images
- Manage timetables and class schedules
- Face security: rejection logs, re-enrollment flows
- Admin access key management

### `campus_vision_engine/` — Rust Vision Workspace

| Crate | Role |
|-------|------|
| `vision_engine` | HTTP API server, CCTV pipeline, camera tracking, face detection, recognition, and liveness verification |
| `intelligence_core` | HNSW-based vector index with SIMD-accelerated cosine similarity; multi-school isolation |
| `intelligence_py` | PyO3 bindings exposing scheduling primitives to Python |

The vision engine is built for **concurrency and throughput** — parallel camera streams, connection pooling, async I/O, and SIMD math. It communicates back to Django to record verified attendance events.

### Frontend Pages

| Page | Description |
|------|-------------|
| `Admin/Dashboard` | Overview metrics, quick actions |
| `Admin/Organizations` | Create & manage orgs and branches |
| `Admin/Timetable` | Visual timetable editor |
| `Admin/Scheduler` | Class scheduling interface |
| `Attendance/AttendanceScanner` | Rolling QR-based check-in (30–40s rotation) |
| `Attendance/AttendanceCamera` | Live camera attendance with three-layer verification |
| `Analytics` | Attendance trends & reports |
| `Students` / `Teachers` | User management & profiles |

---

## Security & Privacy

CampusVision handles biometric data — face images and embeddings. This comes with serious responsibilities:

- **Encryption at rest** — face embeddings stored encrypted in MinIO
- **HTTPS only** — never serve over plain HTTP in production
- **Access Control** — strict role separation (Admin → Teacher → Student)
- **Rolling QR expiry** — server-side token invalidation every 30–40 seconds, no grace period; expired tokens are rejected outright
- **Geo-fencing** — prevents remote/proxy attendance marking
- **Three-layer verification** — face recognition, emotion liveness, and CCTV cross-check before any record is written
- **Audit trails** — `FaceRejectionLog` tracks all failed recognition and verification events
- **Data minimization** — only embeddings (not raw images) used for recognition after enrollment

> Always obtain proper user consent before enrolling biometric data.

---

## Environment Variables

Each service has its own `.env` / `.env.example`. Key variables to configure:

| Service | Key Variables |
|---------|--------------|
| Django Backend | `SECRET_KEY`, `DEBUG`, `DATABASE_URL`, `MINIO_*`, `RUST_ENGINE_URL` |
| Vision Engine | `REDIS_URL`, `MINIO_*`, `DJANGO_CALLBACK_URL`, `MODEL_PATH` |
| Frontend | `VITE_API_BASE_URL` |
| Mobile | `EXPO_PUBLIC_API_URL` |

---

## Testing

### Rust Tests

```bash
# Unit & integration tests
cargo test

# Stress and load tests
./scripts/stress_runner.sh
./scripts/bench_stress_runner.sh
```

Test suites cover: SIMD parity, HNSW search correctness, concurrent inserts, persistence & crash recovery, multi-school isolation, API flows, and failure scenarios (Redis down, model failure, partial crash).

### Django Tests

```bash
cd Backend/attendance_backend
python manage.py test
```

---

## Roadmap

- [ ] Push notifications for attendance alerts (mobile)
- [ ] CSV/Excel export for attendance reports
- [ ] SSO / OAuth2 integration
- [ ] Offline-capable mobile attendance (sync on reconnect)
- [ ] Admin audit log viewer in dashboard

---

## License

This project is **proprietary software**. All rights reserved by Anshu Gondi.
See the [LICENSE](LICENSE) file for full terms.

---

## Author

<div align="center">

**Anshu Gondi**

[GitHub](https://github.com/Anshu-Gondi) · Made with ❤️ and a lot of Rust

</div>
