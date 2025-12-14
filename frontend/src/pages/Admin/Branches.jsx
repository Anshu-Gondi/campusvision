import { useEffect, useState } from "react";
import {
  fetchBranches,
  fetchOrganizations,
  createBranch,
} from "../../services/api";

export default function Branches() {
  const [branches, setBranches] = useState([]);
  const [orgs, setOrgs] = useState([]);
  const [organization, setOrganization] = useState("");
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");

  useEffect(() => {
    fetchOrganizations().then(setOrgs);
    fetchBranches().then(setBranches);
  }, []);

  const create = async () => {
    if (!organization || !name || !location) {
      alert("Organization, Branch name and Location are required");
      return;
    }

    await createBranch({
      organization,
      name,
      location,
    });

    setName("");
    setLocation("");
    fetchBranches().then(setBranches);
  };

  return (
    <>
      <h1 className="title">Branches</h1>

      {/* Create Branch */}
      <div className="box">
        <div className="columns">
          <div className="column">
            <div className="select is-fullwidth">
              <select
                value={organization}
                onChange={(e) => setOrganization(e.target.value)}
              >
                <option value="">Select Organization</option>
                {orgs.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="column">
            <input
              className="input"
              placeholder="Branch Name (Main Campus)"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="column">
            <input
              className="input"
              placeholder="Location (City / Area)"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>

          <div className="column is-2">
            <button className="button is-success is-fullwidth" onClick={create}>
              Create
            </button>
          </div>
        </div>
      </div>

      {/* Branch List */}
      <table className="table is-fullwidth is-striped">
        <thead>
          <tr>
            <th>Organization</th>
            <th>Branch</th>
            <th>Location</th>
          </tr>
        </thead>
        <tbody>
          {branches.map((b) => (
            <tr key={b.id}>
              <td>{b.organization_name}</td>
              <td>{b.name}</td>
              <td>{b.location || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
