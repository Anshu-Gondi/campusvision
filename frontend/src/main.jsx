import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import './index.css'
import App from './App.jsx'
import Students from "./pages/Students/students";
import Teachers from "./pages/Teachers/teachers";
import AttendanceScanner from "./pages/Attendance/AttendanceScanner";
import AttendanceForm from "./pages/Attendance/AttendanceForm";
import AttendanceCamera from "./pages/Attendance/AttendanceCamera";
import Analytics from "./pages/Analytics/analytics";

// Admin Pages
import AdminAccess from "./pages/AdminAccess/AdminAccess";
import AdminLayout from "./pages/Admin/AdminLayout";
import AdminGuard from "./pages/Admin/AdminGuard";
import Dashboard from "./pages/Admin/Dashboard";
import Organizations from "./pages/Admin/Organizations";
import Branches from "./pages/Admin/Branches";
import Timetable from "./pages/Admin/Timetable";

import RegisterOrganization from './pages/RegisterOrganization/RegisterOrganization.jsx';

import 'bulma/css/bulma.min.css';
import "leaflet/dist/leaflet.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/attendance/scan" element={<AttendanceScanner />} />
        <Route path="/attendance/form" element={<AttendanceForm />} />
        <Route path="/attendance/camera" element={<AttendanceCamera />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/register-org" element={<RegisterOrganization />} />
        <Route path="/admin-access" element={<AdminAccess />} />
        <Route
          path="/admin"
          element={
            <AdminGuard>
              <AdminLayout />
            </AdminGuard>
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="orgs" element={<Organizations />} />
          <Route path="branches" element={<Branches />} />
          <Route path="students" element={<Students />} />
          <Route path="teachers" element={<Teachers />} />
          <Route path="timetable" element={<Timetable />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);