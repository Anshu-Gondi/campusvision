import { useEffect, useState } from "react";
import {
  fetchTimetable,
  deleteTimetableEntry,
  uploadTimetable,
  updateTimetableEntry,
  downloadSampleTimetable,
  fetchTeachers, // optional API to get teacher list dynamically
} from "../../services/api";

export default function Timetable() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [previewData, setPreviewData] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  // --- FILTERS ---
  const [filters, setFilters] = useState({
    class_name: "",
    section: "",
    teacher_id: "",
    day_of_week: "",
  });

  const [teachers, setTeachers] = useState([]); // For teacher dropdown

  // --- PAGINATION ---
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [total, setTotal] = useState(0);

  // --- Load teachers for dropdown ---
  useEffect(() => {
    async function loadTeachers() {
      try {
        const teacherList = await fetchTeachers(); // fetch {id, name}
        setTeachers(teacherList);
      } catch (err) {
        console.warn("Failed to load teachers:", err.message);
      }
    }
    loadTeachers();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const paramsObj = {
        class_name: filters.class_name,
        section: filters.section,
        day: filters.day_of_week,
        page,
        page_size: pageSize,
      };

      if (filters.teacher_id) paramsObj.teacher_id = filters.teacher_id;

      const params = new URLSearchParams(paramsObj).toString();
      const res = await fetchTimetable(params);

      setEntries(res.results || []);
      setTotal(res.total || 0);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [filters, page]);

  // --- DELETE ---
  const handleDelete = async (id) => {
    if (!confirm("Delete this entry?")) return;
    try {
      await deleteTimetableEntry(id);
      loadData();
    } catch (err) {
      alert(err.message);
    }
  };

  // --- FILE HANDLERS ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
  };

  const handlePreview = async () => {
    if (!selectedFile) return alert("Select file first");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const preview = await uploadTimetable(formData, true);
      setPreviewData(preview);
      alert(`Preview loaded. Valid rows: ${preview.success_count}`);
    } catch (err) {
      alert(err.message);
    }
  };

  const handleCommit = async () => {
    if (!selectedFile) return alert("Select file first");
    if (!confirm("Commit this timetable to database?")) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await uploadTimetable(formData, false);
      alert(`Committed ${res.success_count} rows`);
      setPreviewData(null);
      setSelectedFile(null);
      loadData();
    } catch (err) {
      alert(err.message);
    }
  };

  // --- UPDATE INLINE ---
  const handleUpdate = async (entry) => {
    const newSubject = prompt("Update subject:", entry.subject);
    if (!newSubject) return;

    try {
      await updateTimetableEntry(entry.id, { subject: newSubject });
      loadData();
    } catch (err) {
      alert(err.message);
    }
  };

  // --- DOWNLOAD SAMPLE ---
  const handleDownloadSample = async () => {
    try {
      const blob = await downloadSampleTimetable();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sample_timetable.xlsx";
      a.click();
    } catch (err) {
      alert(err.message);
    }
  };

  // --- FILTER HANDLER ---
  const handleFilterChange = (e) => {
    setFilters((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
    setPage(1);
  };

  if (loading) return <p>Loading timetable...</p>;

  return (
    <div>
      <h1 className="title has-text-success">Timetable</h1>

      {/* Upload & Preview */}
      <div className="box">
        <input type="file" onChange={handleFileChange} />
        <div className="buttons mt-2">
          <button className="button is-info" onClick={handlePreview}>Preview</button>
          <button className="button is-success" onClick={handleCommit}>Commit</button>
          <button className="button is-warning" onClick={handleDownloadSample}>Download Sample</button>
        </div>
        {previewData && (
          <div className="mt-3">
            <p>Valid Rows: {previewData.success_count}</p>
            <p>Errors: {previewData.results.filter(r => r.status === "fail").length}</p>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="box">
        <h2 className="subtitle">Filters</h2>
        <div className="field is-grouped is-grouped-multiline">
          <div className="control">
            <input
              className="input"
              name="class_name"
              placeholder="Class"
              value={filters.class_name}
              onChange={handleFilterChange}
            />
          </div>
          <div className="control">
            <input
              className="input"
              name="section"
              placeholder="Section"
              value={filters.section}
              onChange={handleFilterChange}
            />
          </div>
          <div className="control">
            <select
              className="input"
              name="teacher_id"
              value={filters.teacher_id}
              onChange={handleFilterChange}
            >
              <option value="">All Teachers</option>
              {teachers.map((t) => (
                <option key={t.id} value={t.id}>{t.id} - {t.name}</option>
              ))}
            </select>
          </div>
          <div className="control">
            <select
              className="input"
              name="day_of_week"
              value={filters.day_of_week}
              onChange={handleFilterChange}
            >
              <option value="">All Days</option>
              {["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Timetable Table */}
      <table className="table is-fullwidth is-striped is-dark">
        <thead>
          <tr>
            <th>Day</th>
            <th>Class</th>
            <th>Section</th>
            <th>Subject</th>
            <th>Teacher</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.id}>
              <td>{e.day_of_week}</td>
              <td>{e.class_name}</td>
              <td>{e.section}</td>
              <td>{e.subject}</td>
              <td>{e.teacher_name}</td>
              <td>{e.start_time} - {e.end_time}</td>
              <td>
                <div className="buttons are-small">
                  <button className="button is-link" onClick={() => handleUpdate(e)}>Edit</button>
                  <button className="button is-danger" onClick={() => handleDelete(e.id)}>Delete</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      <nav className="pagination" role="navigation" aria-label="pagination">
        <button
          className="pagination-previous"
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page === 1}
        >
          Previous
        </button>
        <button
          className="pagination-next"
          onClick={() => setPage((p) => (p * pageSize < total ? p + 1 : p))}
          disabled={page * pageSize >= total}
        >
          Next
        </button>
        <ul className="pagination-list">
          <li>
            <span className="pagination-link is-current">Page {page}</span>
          </li>
        </ul>
      </nav>
    </div>
  );
}