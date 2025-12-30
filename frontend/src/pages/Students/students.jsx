import { useEffect, useState } from "react";
import {
  fetchStudents,
  createStudent,
  updateStudentImage,
  deleteStudentImage,
  deleteStudentFull,
  fetchBranches,
} from "../../services/api";

export default function Students() {
  const [students, setStudents] = useState([]);
  const [branches, setBranches] = useState([]);

  const [form, setForm] = useState({
    name: "",
    roll_no: "",
    class_name: "",
    section: "",
    branch: "",
    image: null,
  });

  const load = async () => {
    setStudents(await fetchStudents());
    setBranches(await fetchBranches());
  };

  useEffect(() => { load(); }, []);

  const submit = async (e) => {
    e.preventDefault();

    // 1️⃣ Create student (JSON)
    const res = await createStudent({
      name: form.name,
      roll_no: form.roll_no,
      class_name: form.class_name,
      section: form.section,
      branch: Number(form.branch),
    });

    // 2️⃣ Upload image (optional)
    if (form.image && res?.id) {
      const fd = new FormData();
      fd.append("image", form.image);
      await updateStudentImage(res.id, fd);
    }

    setForm({
      name: "",
      roll_no: "",
      class_name: "",
      section: "",
      branch: "",
      image: null,
    });

    load();
  };

  return (
    <div className="container">
      <h1 className="title">Admin · Students</h1>

      {/* CREATE */}
      <form onSubmit={submit} className="box">
        <input className="input mb-2" placeholder="Name"
          value={form.name}
          onChange={e => setForm({ ...form, name: e.target.value })}
          required
        />

        <input className="input mb-2" placeholder="Roll No"
          value={form.roll_no}
          onChange={e => setForm({ ...form, roll_no: e.target.value })}
          required
        />

        <input className="input mb-2" placeholder="Class (e.g. 10)"
          value={form.class_name}
          onChange={e => setForm({ ...form, class_name: e.target.value })}
          required
        />

        <input className="input mb-2" placeholder="Section (e.g. A)"
          value={form.section}
          onChange={e => setForm({ ...form, section: e.target.value })}
          required
        />

        {/* BRANCH */}
        <div className="select is-fullwidth mb-2">
          <select
            value={form.branch}
            onChange={e => setForm({ ...form, branch: e.target.value })}
            required
          >
            <option value="">Select Branch</option>
            {branches.map(b => (
              <option key={b.id} value={b.id}>{b.name}</option>
            ))}
          </select>
        </div>

        {/* IMAGE */}
        <input
          type="file"
          accept="image/*"
          className="mb-3"
          onChange={e => setForm({ ...form, image: e.target.files[0] })}
        />

        <button className="button is-success is-fullwidth">
          Create Student
        </button>
      </form>

      {/* LIST */}
      <table className="table is-fullwidth is-striped">
        <thead>
          <tr>
            <th>Name</th>
            <th>Roll</th>
            <th>Class</th>
            <th>Branch</th>
            <th>Image</th>
            <th />
          </tr>
        </thead>

        <tbody>
          {students.map(s => (
            <tr key={s.id}>
              <td>{s.name}</td>
              <td>{s.roll_no}</td>
              <td>{s.class_name}{s.section}</td>
              <td>{s.branch_name}</td>

              <td>
                {s.image && <img src={s.image} width="40" />}
                <input
                  type="file"
                  onChange={async e => {
                    const fd = new FormData();
                    fd.append("image", e.target.files[0]);
                    await updateStudentImage(s.id, fd);
                    load();
                  }}
                />
              </td>

              <td>
                <button className="button is-warning is-small mr-2"
                  onClick={() => deleteStudentImage(s.id)}>
                  Delete Image
                </button>

                <button className="button is-danger is-small"
                  onClick={() => deleteStudentFull(s.id)}>
                  Full Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
