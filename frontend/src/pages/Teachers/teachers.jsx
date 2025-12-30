import { useEffect, useState } from "react";
import {
  fetchTeachers,
  createTeacher,
  updateTeacherImage,
  deleteTeacherImage,
  deleteTeacherFull,
  fetchBranches,
} from "../../services/api";

export default function Teachers() {
  const [teachers, setTeachers] = useState([]);
  const [branches, setBranches] = useState([]);

  const [form, setForm] = useState({
    name: "",
    employee_id: "",
    branch: "",
    department: "",
    subjects: "",
    can_teach_classes: "",
    image: null,
  });

  const load = async () => {
    setTeachers(await fetchTeachers());
    setBranches(await fetchBranches());
  };

  useEffect(() => { load(); }, []);

  const submit = async (e) => {
    e.preventDefault();

    // 1️⃣ Create teacher (JSON)
    const res = await createTeacher({
      name: form.name,
      employee_id: form.employee_id,
      branch: Number(form.branch),
      department: form.department || null,
      subjects: form.subjects.split(",").map(s => s.trim()),
      can_teach_classes: form.can_teach_classes.split(",").map(c => c.trim()),
    });

    // 2️⃣ Upload image if selected
    if (form.image && res?.id) {
      const fd = new FormData();
      fd.append("image", form.image);
      await updateTeacherImage(res.id, fd);
    }

    setForm({
      name: "",
      employee_id: "",
      branch: "",
      department: "",
      subjects: "",
      can_teach_classes: "",
      image: null,
    });

    load();
  };

  const uploadImage = async (id, file) => {
    const fd = new FormData();
    fd.append("image", file);
    await updateTeacherImage(id, fd);
    load();
  };

  return (
    <div className="container">
      <h1 className="title">Admin · Teachers</h1>

      {/* CREATE TEACHER */}
      <form onSubmit={submit} className="box">
        <p className="has-text-grey mb-3">
          Fill teacher details. Fields marked * are required.
        </p>

        <input
          className="input mb-2"
          placeholder="Name * (e.g. Ankit Sharma)"
          value={form.name}
          required
          onChange={e => setForm({ ...form, name: e.target.value })}
        />

        <input
          className="input mb-2"
          placeholder="Employee ID * (e.g. EMP1023)"
          value={form.employee_id}
          required
          onChange={e => setForm({ ...form, employee_id: e.target.value })}
        />

        <div className="select is-fullwidth mb-2">
          <select
            required
            value={form.branch}
            onChange={e => setForm({ ...form, branch: e.target.value })}
          >
            <option value="">Select Branch *</option>
            {branches.map(b => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        </div>

        <input
          className="input mb-2"
          placeholder="Department ID (optional, e.g. 3)"
          value={form.department}
          onChange={e => setForm({ ...form, department: e.target.value })}
        />

        <input
          className="input mb-2"
          placeholder="Subjects (comma separated) e.g. Math, Physics"
          value={form.subjects}
          onChange={e => setForm({ ...form, subjects: e.target.value })}
        />

        <input
          className="input mb-2"
          placeholder="Classes (comma separated) e.g. 10A, 10B, 11C"
          value={form.can_teach_classes}
          onChange={e => setForm({ ...form, can_teach_classes: e.target.value })}
        />

        {/* IMAGE UPLOAD */}
        <div className="file mb-3">
          <label className="file-label">
            <input
              className="file-input"
              type="file"
              accept="image/*"
              onChange={e => setForm({ ...form, image: e.target.files[0] })}
            />
            <span className="file-cta">
              <span className="file-label">
                Upload Teacher Photo (optional)
              </span>
            </span>
          </label>
        </div>

        <button className="button is-success is-fullwidth">
          Create Teacher
        </button>
      </form>

      {/* LIST TEACHERS */}
      <table className="table is-fullwidth is-striped">
        <thead>
          <tr>
            <th>Name</th>
            <th>Employee</th>
            <th>Branch</th>
            <th>Subjects</th>
            <th>Classes</th>
            <th>Image</th>
            <th>Actions</th>
          </tr>
        </thead>

        <tbody>
          {teachers.map(t => (
            <tr key={t.id}>
              <td>{t.name}</td>
              <td>{t.employee_id}</td>
              <td>{t.branch_name}</td>
              <td>{t.subjects?.join(", ")}</td>
              <td>{t.can_teach_classes?.join(", ")}</td>

              <td>
                {t.image && (
                  <img
                    src={t.image}
                    alt="teacher"
                    width="40"
                    style={{ borderRadius: "6px" }}
                  />
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={e => uploadImage(t.id, e.target.files[0])}
                />
              </td>

              <td>
                <button
                  className="button is-warning is-small mr-2"
                  onClick={() => deleteTeacherImage(t.id)}
                >
                  Delete Image
                </button>

                <button
                  className="button is-danger is-small"
                  onClick={() => deleteTeacherFull(t.id)}
                >
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
