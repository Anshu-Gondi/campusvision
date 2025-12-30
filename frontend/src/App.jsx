import { Routes, Route, Link } from "react-router-dom";
import Students from "./pages/Students/students";
import Teachers from "./pages/Teachers/teachers";
import AttendanceScanner from "./pages/Attendance/AttendanceScanner";
import Analytics from "./pages/Analytics/analytics";
import Home from "./pages/Home/home";

import "./App.css";

function App() {
  return (
    <>
      {/* Navbar */}
      <nav className="navbar is-dark" role="navigation" aria-label="main navigation">
        <div className="navbar-brand">
          <Link className="navbar-item" to="/">CampusVision</Link>
        </div>
        <div className="navbar-menu">
          <div className="navbar-start">
            <Link className="navbar-item" to="/students">Students</Link>
            <Link className="navbar-item" to="/teachers">Teachers</Link>
            <Link className="navbar-item" to="/attendance/scan">Attendance</Link>
            <Link className="navbar-item" to="/analytics">Analytics</Link>
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <section className="section">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/students" element={<Students />} />
            <Route path="/teachers" element={<Teachers />} />
            <Route path="/attendance/scan" element={<AttendanceScanner />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </div>
      </section>
    </>
  );
}

export default App;
