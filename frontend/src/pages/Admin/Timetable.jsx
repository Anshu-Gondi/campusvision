import { useEffect, useState } from "react";
import {
  fetchTimetable,
  deleteTimetableEntry,
  uploadTimetable,
  updateTimetableEntry,
  downloadSampleTimetable,
} from "../../services/api";

export default function Timetable() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [previewData, setPreviewData] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const loadData = async () => {
    try {
      const res = await fetchTimetable();
      setEntries(res.results || []);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // ================= DELETE =================
  const handleDelete = async (id) => {
    if (!confirm("Delete this entry?")) return;

    try {
      await deleteTimetableEntry(id);
      loadData();
    } catch (err) {
      alert(err.message);
    }
  };

  // ================= FILE SELECT =================
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
  };

  // ================= PREVIEW =================
  const handlePreview = async () => {
    if (!selectedFile) {
      alert("Select file first");
      return;
    }

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

  // ================= COMMIT =================
  const handleCommit = async () => {
    if (!selectedFile) {
      alert("Select file first");
      return;
    }

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

  // ================= UPDATE (Quick Inline) =================
  const handleUpdate = async (entry) => {
    const newSubject = prompt("Update subject:", entry.subject);
    if (!newSubject) return;

    try {
      await updateTimetableEntry(entry.id, {
        subject: newSubject,
      });
      loadData();
    } catch (err) {
      alert(err.message);
    }
  };

  // ================= DOWNLOAD SAMPLE =================
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

  if (loading) return <p>Loading timetable...</p>;

  return (
    <div>
      <h1 className="title has-text-success">Timetable</h1>

      {/* Upload Section */}
      <div className="box">
        <div className="field">
          <input type="file" onChange={handleFileChange} />
        </div>

        <div className="buttons mt-2">
          <button className="button is-info" onClick={handlePreview}>
            Preview
          </button>

          <button className="button is-success" onClick={handleCommit}>
            Commit
          </button>

          <button className="button is-warning" onClick={handleDownloadSample}>
            Download Sample
          </button>
        </div>

        {previewData && (
          <div className="mt-3">
            <p>Valid Rows: {previewData.success_count}</p>
            <p>Errors: {previewData.error_count}</p>
          </div>
        )}
      </div>

      {/* Table */}
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
              <td>
                {e.start_time} - {e.end_time}
              </td>
              <td>
                <div className="buttons are-small">
                  <button
                    className="button is-link"
                    onClick={() => handleUpdate(e)}
                  >
                    Edit
                  </button>
                  <button
                    className="button is-danger"
                    onClick={() => handleDelete(e.id)}
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}