import { Link, Outlet } from "react-router-dom";

export default function AdminLayout() {
  return (
    <>
      <nav className="navbar is-black">
        <div className="navbar-brand">
          <span className="navbar-item has-text-success">🔐 Admin Control</span>
        </div>

        <div className="navbar-menu">
          <div className="navbar-start">
            <Link className="navbar-item" to="/admin">
              Dashboard
            </Link>
            <Link className="navbar-item" to="/admin/orgs">
              Organizations
            </Link>
            <Link className="navbar-item" to="/admin/branches">
              Branches
            </Link>
            <Link className="navbar-item" to="/admin/students">
              Students
            </Link>
            <Link className="navbar-item" to="/admin/teachers">
              Teachers
            </Link>
          </div>

          <div className="navbar-end">
            <button
              className="button is-danger is-small m-2"
              onClick={() => {
                localStorage.removeItem("admin_token");
                window.location.href = "/";
              }}
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      <section className="section">
        <div className="container">
          <Outlet />
        </div>
      </section>
    </>
  );
}
