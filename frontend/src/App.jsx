import { Routes, Route, Link } from "react-router-dom";
import { useState } from "react";

import Home from "./pages/Home/home";
import RegisterOrganization from "./pages/RegisterOrganization/RegisterOrganization";
import AdminAccess from "./pages/AdminAccess/AdminAccess";

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
      {/* NAVBAR */}
      <nav className="navbar is-dark main-navbar" role="navigation">
        <div className="navbar-brand">
          <Link className="navbar-item brand" to="/" onClick={closeNavbar}>
            CampusVision
          </Link>

          <button
            className={`navbar-burger ${isActive ? "is-active" : ""}`}
            onClick={toggleNavbar}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
        </div>

        <div className={`navbar-menu ${isActive ? "is-active" : ""}`}>
          <div className="navbar-end">

            {/* Secondary */}
            <Link
              className="navbar-item admin-link"
              to="/admin-access"
              onClick={closeNavbar}
            >
              Admin Access
            </Link>

            {/* Primary CTA */}
            <Link
              className="button is-success register-btn"
              to="/register-org"
              onClick={closeNavbar}
            >
              Register Institution
            </Link>

          </div>
        </div>
      </nav>

      {/* PAGE */}
      <section className="section">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/register-org" element={<RegisterOrganization />} />
            <Route path="/admin-access" element={<AdminAccess />} />
          </Routes>
        </div>
      </section>
    </>
  );
}

export default App;