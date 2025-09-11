import { useEffect, useState } from "react";
import { fetchStudents, createStudent, updateStudentImage, deleteStudent } from "../../services/api";

export default function Students() {
  const [students, setStudents] = useState([]);
  const [form, setForm] = useState({ name: "", roll_no: "", class_name: "", section: "", image: null });
  const [loading, setLoading] = useState(false);

  const loadStudents = async () => {
    setLoading(true);
    const data = await fetchStudents();
    setStudents(data);
    setLoading(false);
  };

  useEffect(() => {
    loadStudents();
  }, []);

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setForm({ ...form, [name]: files ? files[0] : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    for (const key in form) formData.append(key, form[key]);
    await createStudent(formData);
    setForm({ name: "", roll_no: "", class_name: "", section: "", image: null });
    loadStudents();
  };

  const handleImageUpdate = async (studentId, file) => {
    const formData = new FormData();
    formData.append("image", file);
    await updateStudentImage(studentId, formData);
    loadStudents();
  };

  const handleDelete = async (rollNo) => {
    await deleteStudent(rollNo);
    loadStudents();
  };

  return (
    <div className="container">
      <h1 className="title has-text-light">Students Management</h1>

      <form onSubmit={handleSubmit} className="box">
        <div className="field">
          <label className="label">Name</label>
          <input className="input" name="name" value={form.name} onChange={handleChange} required />
        </div>
        <div className="field">
          <label className="label">Roll No</label>
          <input className="input" name="roll_no" value={form.roll_no} onChange={handleChange} required />
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
        <button className="button is-success mt-3" type="submit">Add Student</button>
      </form>

      <h2 className="subtitle mt-5">Student Records</h2>
      {loading ? <p>Loading...</p> : (
        <table className="table is-fullwidth is-striped is-hoverable">
          <thead>
            <tr>
              <th>Name</th>
              <th>Roll No</th>
              <th>Class</th>
              <th>Section</th>
              <th>Image</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {students.map(student => (
              <tr key={student.id}>
                <td>{student.name}</td>
                <td>{student.roll_no}</td>
                <td>{student.class_name}</td>
                <td>{student.section}</td>
                <td>
                  {student.image ? <img src={student.image} alt={student.name} style={{ width: "50px" }} /> : "No image"}
                  <input type="file" onChange={(e) => handleImageUpdate(student.id, e.target.files[0])} className="mt-1" />
                </td>
                <td>
                  <button className="button is-danger is-small" onClick={() => handleDelete(student.roll_no)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
