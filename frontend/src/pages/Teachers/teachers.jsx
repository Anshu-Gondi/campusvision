import { useEffect, useState } from "react";
import {
  fetchTeachers,
  createTeacher,
  updateTeacherImage,
  deleteTeacher,
  deleteTeacherFull,   // 🆕 import full delete
} from "../../services/api";

export default function Teachers() {
  const [teachers, setTeachers] = useState([]);
  const [form, setForm] = useState({
    name: "",
    employee_id: "",
    subject: "",
    class_name: "",
    section: "",
    image: null,
  });
  const [loading, setLoading] = useState(false);

  const loadTeachers = async () => {
    setLoading(true);
    const data = await fetchTeachers();
    setTeachers(data);
    setLoading(false);
  };

  useEffect(() => {
    loadTeachers();
  }, []);

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setForm({ ...form, [name]: files ? files[0] : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    for (const key in form) formData.append(key, form[key]);
    await createTeacher(formData);
    setForm({
      name: "",
      employee_id: "",
      subject: "",
      class_name: "",
      section: "",
      image: null,
    });
    loadTeachers();
  };

  const handleImageUpdate = async (teacherId, file) => {
    const formData = new FormData();
    formData.append("image", file);
    await updateTeacherImage(teacherId, formData);
    loadTeachers();
  };

  // 🟢 Old: delete only image
  const handleDeleteImage = async (employeeId) => {
    await deleteTeacher(employeeId);
    loadTeachers();
  };

  // 🆕 New: full delete (teacher + attendance + image)
  const handleDeleteFull = async (employeeId) => {
    if (window.confirm("This will permanently delete teacher, image, and attendance records. Continue?")) {
      await deleteTeacherFull(employeeId);
      loadTeachers();
    }
  };

  return (
    <div className="container">
      <h1 className="title has-text-light">Teachers Management</h1>

      <form onSubmit={handleSubmit} className="box">
        <div className="field">
          <label className="label">Name</label>
          <input className="input" name="name" value={form.name} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Employee ID</label>
          <input className="input" name="employee_id" value={form.employee_id} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Subject</label>
          <input className="input" name="subject" value={form.subject} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Class</label>
          <input className="input" name="class_name" value={form.class_name} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Section</label>
          <input className="input" name="section" value={form.section} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Image</label>
          <input className="input" type="file" name="image" onChange={handleChange} />
        </div>
        <button className="button is-success mt-3" type="submit">Add Teacher</button>
      </form>

      <h2 className="subtitle mt-5">Teacher Records</h2>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <table className="table is-fullwidth is-striped is-hoverable">
          <thead>
            <tr>
              <th>Name</th>
              <th>Employee ID</th>
              <th>Subject</th>
              <th>Class</th>
              <th>Section</th>
              <th>Image</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {teachers.map((teacher) => (
              <tr key={teacher.id}>
                <td>{teacher.name}</td>
                <td>{teacher.employee_id}</td>
                <td>{teacher.subject}</td>
                <td>{teacher.class_name}</td>
                <td>{teacher.section}</td>
                <td>
                  {teacher.image_url ? (
                    <img
                      src={teacher.image_url}
                      alt={teacher.name}
                      style={{ width: "50px" }}
                    />
                  ) : (
                    "No image"
                  )}
                  <input
                    type="file"
                    onChange={(e) => handleImageUpdate(teacher.id, e.target.files[0])}
                    className="mt-1"
                  />
                </td>
                <td>
                  <button
                    className="button is-warning is-small mr-2"
                    onClick={() => handleDeleteImage(teacher.employee_id)}
                  >
                    Delete Image
                  </button>
                  <button
                    className="button is-danger is-small"
                    onClick={() => handleDeleteFull(teacher.employee_id)}
                  >
                    Full Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
