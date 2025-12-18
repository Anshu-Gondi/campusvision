import { useEffect, useState } from "react";
import {
  fetchBranches,
  getOrganization,
  createBranch,
  updateBranch,
  deleteBranch,
} from "../../services/api";

export default function Branches() {
  const [branches, setBranches] = useState([]);
  const [orgs, setOrgs] = useState([]);
  const [organization, setOrganization] = useState("");
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [editingId, setEditingId] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setOrgs(await getOrganization());
    setBranches(await fetchBranches());
  };

  const resetForm = () => {
    setOrganization("");
    setName("");
    setLocation("");
    setEditingId(null);
  };

  const submit = async () => {
    if (!organization || !name) {
      alert("Organization and Branch name required");
      return;
    }

    const payload = { organization, name, location };

    if (editingId) {
      await updateBranch(editingId, payload);
    } else {
      await createBranch(payload);
    }

    resetForm();
    loadData();
  };

  const edit = (b) => {
    setEditingId(b.id);
    setOrganization(b.organization);
    setName(b.name);
    setLocation(b.location || "");
  };

  const remove = async (id) => {
    if (!window.confirm("Delete this branch?")) return;
    await deleteBranch(id);
    loadData();
  };

  return (
    <>
      <h1 className="title">Branches</h1>

      {/* Create / Edit */}
      <div className="box">
        <div className="columns is-multiline">
          <div className="column is-4">
            <div className="select is-fullwidth">
              <select value={organization} onChange={(e) => setOrganization(e.target.value)}>
                <option value="">Select Organization</option>
                {orgs.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Branch Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>

          <div className="column is-2">
            <button className="button is-success is-fullwidth" onClick={submit}>
              {editingId ? "Update" : "Create"}
            </button>
          </div>
        </div>
      </div>

      {/* Branch Table */}
      <table className="table is-fullwidth is-striped">
        <thead>
          <tr>
            <th>Organization</th>
            <th>Branch</th>
            <th>Location</th>
            <th width="180">Actions</th>
          </tr>
        </thead>
        <tbody>
          {branches.map((b) => (
            <tr key={b.id}>
              <td>{b.organization_name}</td>
              <td>{b.name}</td>
              <td>{b.location || "—"}</td>
              <td>
                <button className="button is-small is-info mr-2" onClick={() => edit(b)}>
                  Edit
                </button>
                <button className="button is-small is-danger" onClick={() => remove(b.id)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
