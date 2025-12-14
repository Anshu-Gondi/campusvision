import { useEffect, useState } from "react";
import { fetchOrganizations, createOrganization } from "../../services/api";

export default function Organizations() {
  const [orgs, setOrgs] = useState([]);
  const [name, setName] = useState("");
  const [type, setType] = useState("school");

  useEffect(() => {
    fetchOrganizations().then(setOrgs);
  }, []);

  const create = async () => {
    await createOrganization({ name, org_type: type });
    setName("");
    fetchOrganizations().then(setOrgs);
  };

  return (
    <>
      <h1 className="title">Organizations</h1>

      <div className="box">
        <input className="input mb-2" placeholder="Name" value={name} onChange={e => setName(e.target.value)} />
        <div className="select mb-2">
          <select value={type} onChange={e => setType(e.target.value)}>
            <option value="school">School</option>
            <option value="college">College</option>
          </select>
        </div>
        <button className="button is-success" onClick={create}>Create</button>
      </div>

      <table className="table is-fullwidth">
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {orgs.map(o => (
            <tr key={o.id}>
              <td>{o.name}</td>
              <td>{o.org_type}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
