import { useState } from "react";

function RegisterOrganization() {
  const [form, setForm] = useState({
    org_name: "",
    org_type: "school",
    website: "",
    branch_name: "",
    city: "",
    state: ""
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const res = await fetch("http://localhost:8000/api/register-organization/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="container">
      <h1 className="title is-3">Register Your Organization</h1>

      <form onSubmit={submit} className="box">
        <div className="field">
          <label className="label">Organization Name</label>
          <input className="input" name="org_name" required onChange={handleChange} />
        </div>

        <div className="field">
          <label className="label">Organization Type</label>
          <div className="select is-fullwidth">
            <select name="org_type" onChange={handleChange}>
              <option value="school">School</option>
              <option value="college">College</option>
              <option value="institute">Institute</option>
            </select>
          </div>
        </div>

        <div className="field">
          <label className="label">Website (optional)</label>
          <input className="input" name="website" onChange={handleChange} />
        </div>

        <hr />

        <div className="field">
          <label className="label">Branch Name</label>
          <input className="input" name="branch_name" required onChange={handleChange} />
        </div>

        <div className="field is-grouped">
          <input className="input" placeholder="City" name="city" onChange={handleChange} />
          <input className="input ml-2" placeholder="State" name="state" onChange={handleChange} />
        </div>

        <button className={`button is-primary mt-4 ${loading ? "is-loading" : ""}`}>
          Register Organization
        </button>
      </form>

      {result?.default_admin_password && (
        <article className="message is-warning mt-4">
          <div className="message-header">
            <p>Admin Access Created</p>
          </div>
          <div className="message-body">
            <p><strong>Default Admin Password:</strong></p>
            <h2 className="title is-4">{result.default_admin_password}</h2>
            <p>⚠ Save this now. You must change it after login.</p>
          </div>
        </article>
      )}
    </div>
  );
}

export default RegisterOrganization;
