import { useEffect, useState } from "react";
import {
  getOrganization,
  updateOrganization,
  rotateAdminCombo,
} from "../../services/api";

export default function Organizations() {
  const [org, setOrg] = useState(null);
  const [orgType, setOrgType] = useState("");
  const [website, setWebsite] = useState("");
  const [newCombo, setNewCombo] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    loadOrg();
  }, []);

  const loadOrg = async () => {
    const data = await getOrganization();
    setOrg(data);
    setOrgType(data.org_type);
    setWebsite(data.website || "");
  };

  const saveOrg = async () => {
    await updateOrganization({
      org_type: orgType,
      website,
    });
    setMessage("✅ Organization updated");
    loadOrg();
  };

  const rotateCombo = async () => {
    if (newCombo.length < 6) {
      setMessage("❌ Combo too short");
      return;
    }

    await rotateAdminCombo({ combo: newCombo });
    setNewCombo("");
    setMessage("🔐 Admin combo rotated");
  };

  if (!org) return <p>Loading...</p>;

  return (
    <>
      <h1 className="title">Organization Settings</h1>

      {/* === ORG INFO === */}
      <div className="box">
        <p><strong>Name:</strong> {org.name}</p>
        <p><strong>Created:</strong> {new Date(org.created_at).toLocaleString()}</p>
      </div>

      {/* === UPDATE ORG === */}
      <div className="box">
        <h2 className="subtitle">Organization Details</h2>

        <div className="field">
          <label className="label">Type</label>
          <div className="select">
            <select value={orgType} onChange={e => setOrgType(e.target.value)}>
              <option value="school">School</option>
              <option value="college">College</option>
              <option value="company">Company</option>
            </select>
          </div>
        </div>

        <div className="field">
          <label className="label">Website</label>
          <input
            className="input"
            value={website}
            onChange={e => setWebsite(e.target.value)}
            placeholder="https://example.com"
          />
        </div>

        <button className="button is-primary" onClick={saveOrg}>
          Save Changes
        </button>
      </div>

      {/* === SECURITY === */}
      <div className="box">
        <h2 className="subtitle">Admin Security</h2>

        <input
          className="input mb-2"
          placeholder="New admin combo"
          value={newCombo}
          onChange={e => setNewCombo(e.target.value)}
        />

        <button className="button is-danger" onClick={rotateCombo}>
          Rotate Admin Combo
        </button>
      </div>

      {message && <p className="has-text-success mt-3">{message}</p>}
    </>
  );
}
