import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

export default function AttendanceForm() {
  const navigate = useNavigate();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const qrCode = queryParams.get("code");

  const [form, setForm] = useState({ type: "student", id: "" });

  const handleFormSubmit = (e) => {
    e.preventDefault();
    // Pass form data and qrCode to camera page
    navigate("/attendance/camera", { state: { form, qrCode } });
  };

  return (
    <div className="container has-text-centered">
      <h1 className="title has-text-primary">Attendance Form</h1>
      <form onSubmit={handleFormSubmit} className="box">
        <div className="field">
          <label className="label">Who are you?</label>
          <div className="control">
            <div className="select">
              <select
                value={form.type}
                onChange={(e) => setForm({ ...form, type: e.target.value })}
              >
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
              </select>
            </div>
          </div>
        </div>

        <div className="field">
          <label className="label">
            {form.type === "student" ? "Roll Number" : "Employee ID"}
          </label>
          <input
            className="input"
            value={form.id}
            onChange={(e) => setForm({ ...form, id: e.target.value })}
            required
          />
        </div>

        <button className="button is-primary" type="submit">
          Proceed to Face Capture
        </button>
      </form>
    </div>
  );
}
