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

import 'bulma/css/bulma.min.css';

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/students" element={<Students />} />
        <Route path="/teachers" element={<Teachers />} />
        <Route path="/attendance/scan" element={<AttendanceScanner />} />
        <Route path="/attendance/form" element={<AttendanceForm />} />
        <Route path="/attendance/camera" element={<AttendanceCamera />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);