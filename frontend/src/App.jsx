import { Routes, Route, Link } from "react-router-dom";
import { useState } from "react";

import Students from "./pages/Students/students";
import Teachers from "./pages/Teachers/teachers";
import AttendanceScanner from "./pages/Attendance/AttendanceScanner";
import Analytics from "./pages/Analytics/analytics";
import Home from "./pages/Home/home";

import "./App.css";

function App() {
  const [isActive, setIsActive] = useState(false);

  const toggleNavbar = () => {
    setIsActive(!isActive);
  };

  const closeNavbar = () => {
    setIsActive(false);
  };

  return (
    <>
      {/* Navbar */}
      <nav className="navbar is-dark" role="navigation">
        <div className="navbar-brand">
          <Link className="navbar-item" to="/" onClick={closeNavbar}>
            CampusVision
          </Link>

          {/* Burger Menu (Mobile) */}
          <button
            className={`navbar-burger ${isActive ? "is-active" : ""}`}
            aria-label="menu"
            aria-expanded="false"
            onClick={toggleNavbar}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
        </div>

        <div className={`navbar-menu ${isActive ? "is-active" : ""}`}>
          <div className="navbar-start">
            <Link className="navbar-item" to="/students" onClick={closeNavbar}>
              Students
            </Link>
            <Link className="navbar-item" to="/teachers" onClick={closeNavbar}>
              Teachers
            </Link>
            <Link className="navbar-item" to="/attendance/scan" onClick={closeNavbar}>
              Attendance
            </Link>
            <Link className="navbar-item" to="/analytics" onClick={closeNavbar}>
              Analytics
            </Link>
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