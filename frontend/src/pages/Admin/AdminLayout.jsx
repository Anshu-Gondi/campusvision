import { NavLink, Outlet } from "react-router-dom";
import { useState } from "react";

export default function AdminLayout() {
  const [isActive, setIsActive] = useState(false);

  const toggleSidebar = () => {
    setIsActive(!isActive);
  };

  const closeSidebar = () => {
    setIsActive(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("admin_token");
    window.location.href = "/";
  };

  const linkClass = ({ isActive }) =>
    isActive ? "sidebar-link active" : "sidebar-link";

  return (
    <div className="admin-layout">
      <aside className={`admin-sidebar ${isActive ? "is-active" : ""}`}>
        <div className="sidebar-header">
          <h2>🔐 Admin Control</h2>
        </div>

        <nav className="menu">
          <ul className="menu-list">
            <li>
              <NavLink to="/admin" end className={linkClass} onClick={closeSidebar}>
                Dashboard
              </NavLink>
            </li>
            <li>
              <NavLink to="/admin/orgs" className={linkClass} onClick={closeSidebar}>
                Organizations
              </NavLink>
            </li>
            <li>
              <NavLink to="/admin/branches" className={linkClass} onClick={closeSidebar}>
                Branches
              </NavLink>
            </li>
            <li>
              <NavLink to="/admin/students" className={linkClass} onClick={closeSidebar}>
                Students
              </NavLink>
            </li>
            <li>
              <NavLink to="/admin/teachers" className={linkClass} onClick={closeSidebar}>
                Teachers
              </NavLink>
            </li>
            <li>
              <NavLink to="/admin/timetable" className={linkClass} onClick={closeSidebar}>
                Timetable
              </NavLink>
            </li>
          </ul>
        </nav>

        <div className="sidebar-footer">
          <button
            className="button is-danger is-small is-fullwidth"
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>
      </aside>

      <button
        className="sidebar-toggle button is-black is-hidden-desktop"
        onClick={toggleSidebar}
      >
        ☰
      </button>

      {isActive && <div className="admin-overlay" onClick={closeSidebar}></div>}

      <main className="admin-content" onClick={closeSidebar}>
        <Outlet />
      </main>
    </div>
  );
}