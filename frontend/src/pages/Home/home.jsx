import { Link } from "react-router-dom";
import "./style.css"; // import our glowing theme

function Home() {
  return (
    <section className="hero is-fullheight is-dark is-bold">
      <div className="hero-body">
        <div className="container has-text-centered">
          <h1 className="title is-1">✨ Welcome to Attendance System ✨</h1>
          <h2 className="subtitle is-4">Manage Students, Teachers, and Attendance with Ease</h2>

          <div className="columns is-centered mt-6">
            {/* Register Records */}
            <div className="column is-4">
              <div className="card">
                <div className="card-content">
                  <p className="title is-4">Register Records</p>
                  <p className="subtitle is-6">Add Students or Teachers</p>
                  <Link to="/students" className="button is-primary is-fullwidth mb-2">
                    Student Records
                  </Link>
                  <Link to="/teachers" className="button is-link is-fullwidth">
                    Teacher Records
                  </Link>
                </div>
              </div>
            </div>

            {/* Start Attendance */}
            <div className="column is-4">
              <div className="card">
                <div className="card-content">
                  <p className="title is-4">Start Attendance</p>
                  <p className="subtitle is-6">Mark Daily Attendance Quickly</p>
                  <Link to="/attendance/scan" className="button is-success is-fullwidth">
                    Take Attendance
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Analytics */}
          <div className="mt-5">
            <Link to="/analytics" className="button is-warning is-medium">
              View Analytics
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Home;
