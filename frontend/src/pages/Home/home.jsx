import { Link } from "react-router-dom";
import "./style.css";

function Home() {
  return (
    <>
      {/* HERO */}
      <section className="hero is-fullheight is-dark is-bold">
        <div className="hero-body">
          <div className="container has-text-centered">
            <h1 className="title is-1">
              Stop Proxy Attendance. Automate It.
            </h1>

            <h2 className="subtitle is-4 mt-4">
              Face Recognition + Location Verification + CCTV Validation.
              Integrated with a dedicated mobile app for attendance capture.
              Designed for institutions that need fraud-resistant attendance systems.
            </h2>

            <div className="mt-6">
              <Link to="/register-org" className="button is-success register-cta">
                Register Your Institution
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="section clean-section">
        <div className="container has-text-centered">
          <h1 className="title is-2 section-title">How CampusVision Works</h1>

          <div className="columns mt-6 is-variable is-5">
            <div className="column">
              <div className="feature-card">
                <h3 className="title is-5">Face Recognition</h3>
                <p>
                  Attendance is marked instantly using face scan.
                  No manual entry. No proxy.
                </p>
              </div>
            </div>

            <div className="column">
              <div className="feature-card">
                <h3 className="title is-5">Location Validation</h3>
                <p>
                  System verifies presence inside campus.
                  Attempts from outside are automatically flagged.
                </p>
              </div>
            </div>

            <div className="column">
              <div className="feature-card">
                <h3 className="title is-5">CCTV Verification</h3>
                <p>
                  Remote entries are validated using CCTV within
                  a controlled time window.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ADMIN CONTROL */}
      <section className="section has-background-dark">
        <div className="container has-text-centered">
          <h1 className="title is-2">Built for Institutions</h1>

          <p className="subtitle is-5 mt-4">
            Control multiple branches, manage timetables,
            and monitor attendance analytics — all from one system.
          </p>

          <div className="mt-5">
            <Link to="/admin-access" className="button is-primary is-medium">
              Admin Access
            </Link>
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="section has-background-black">
        <div className="container has-text-centered">
          <h1 className="title is-3">
            Secure Your Attendance System Today
          </h1>

          <div className="mt-5">
            <Link to="/register-org" className="button is-success is-medium">
              Get Started
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

export default Home;