# CampusVision — Face Recognition Attendance for Institutions

CampusVision is an advanced, production-ready attendance platform for educational and corporate organizations. It combines automated face recognition with management dashboards to modernize and secure attendance workflows for students, teachers, and administrators.

---

## Features

- **Automatic Attendance:** Uses face recognition (OpenCV, etc.) to mark attendance for students and teachers.
- **Multiple Interfaces:**
  - **Admin Web Dashboard:** Manage classes, schedules, reports, organizations.
  - **Teacher/Student Mobile App:** Check in/out, view records, receive notifications.
- **Organization Management:** Admins can bulk upload, schedule timetables, manage roles via dedicated modules.
- **Live & Image Modes:** Real-time via CCTV/camera (`campus_vision_engine/vision_engine`), or via image uploads.
- **Secure & Auditable:** User authentication, access control, audit trails.

---

## Technology Stack

| Layer        | Technology                       |
| ------------ | ------------------------------- |
| Backend      | Python, Django                   |
| Face Engine  | Python (OpenCV, dlib), Rust (optional for perf), custom modules (`campus_vision_engine`) |
| Frontend     | React (Vite), HTML/CSS/JS        |
| Mobile       | React Native (Expo, TypeScript)  |
| Database     | Django ORM (default: SQLite, prod: PostgreSQL/MySQL/etc.) |

---

## Project Structure

```
campusvision/
│
├── Backend/
│   └── attendance_backend/
│         ├── attendance/              # Attendance logic: models, APIs, serializers, utils, views
│         ├── Admin/                   # Organization management: admin workflows, bulk upload, timetable, face security
│         ├── attendance_backend/      # Django config: settings, urls, wsgi/asgi
│         ├── campus_vision_engine/    # Face recognition & vision modules (intelligence_core, vision_engine, etc)
│         ├── manage.py, requirements.txt, etc
├── frontend/
│     └── src/
│         ├── pages/                   # UI pages (Admin, Attendance, Analytics, Students, Teachers, etc.)
│         ├── services/                # API layer (api.js)
├── mobile/
│     ├── components/                  # Shared React Native UI components
│     ├── hooks/                       # Custom mobile React hooks
│     ├── constants/                   # Theme and config
│     ├── app/                         # Main app logic and navigation
├── docker/                            # Deployment configs (docker-compose)
└── README.md, LICENSE, etc.
```

---

## Setup & Installation

### 1. Backend

- Clone the repo, create a Python venv, install dependencies:
  ```bash
  cd Backend/attendance_backend
  python -m venv venv
  source venv/bin/activate
  pip install -r ../requirements.txt
  python manage.py migrate
  python manage.py runserver
  ```

- For face recognition, dependencies include OpenCV, dlib, and possibly Rust components.

### 2. Frontend (Admin Dashboard)

```bash
cd frontend
pnpm install       # or npm/yarn install
pnpm dev
```

### 3. Mobile (React Native Expo App)

```bash
cd mobile
pnpm install
pnpm start
```
> Scan QR with Expo Go, or use Android/iOS simulator.

---

## Core Modules Breakdown

### Backend: `attendance/`
- Registers, verifies attendance, manages attendance records, schedules.
- APIs: `/attendance/mark/`, `/attendance/report/`, etc.
- Models: Student, Teacher, AttendanceRecord, etc.

### Backend: `Admin/`
- Organization/user management, bulk upload, schedule/timetable, face security handling.
- APIs: `/admin/upload/`, `/admin/timetable/`, `/admin/users/`

### Backend: `campus_vision_engine/`
- Face recognition engine, hooks into camera/webcam or image upload flow.

### Frontend
- `src/pages/Attendance` — mark/view attendance.
- `src/pages/Admin` — dashboards and admin functions.
- `src/pages/Analytics`, `Students`, `Teachers` — reporting and user management.

### Mobile
- Home, Attendance, Profile tabs/screens, reusable components (`components/`), hooks for color/theme.

---

## Configuration

- **.env and secrets:** Set environment variables for production (database, API security keys, etc.)
- **Database:** SQLite by default (see Django settings), switch to PostgreSQL/MySQL for production.

---

## Security & Privacy

- Biometric data is sensitive: use encryption and consent flows.
- HTTPS only in production.
- Regularly update dependencies for security patches.

---

## License

This repository and code are proprietary to Anshu-Gondi. See LICENSE file for terms.

---

## Author

Made with ❤️ by [Anshu Gondi](https://github.com/Anshu-Gondi)