import { useEffect, useState, useCallback } from "react";
import {
  fetchStudents,
  createStudent,
  updateStudentImage,
  deleteStudentImage,
  deleteStudentFull,
  fetchBranches,
  uploadStudentsExcel,
  downloadSampleStudentsExcel
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

  const [bulkFile, setBulkFile] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [bulkLoading, setBulkLoading] = useState(false);

  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [count, setCount] = useState(0);
  const pageSize = 10;

  const [filters, setFilters] = useState({
    name: "",
    roll_no: "",
    class_name: "",
    branch: "",
  });

  const [debouncedFilters, setDebouncedFilters] = useState(filters);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
      setCurrentPage(1);
    }, 400);

    return () => clearTimeout(timer);
  }, [filters]);
  const loadStudents = useCallback(async (page = 1) => {
    const params = {
      page,
      page_size: pageSize,
    };

    if (debouncedFilters.name) {
      params.name = debouncedFilters.name;
    }

    if (debouncedFilters.roll_no) {
      params.roll_no = debouncedFilters.roll_no;
    }

    if (debouncedFilters.class_name) {
      params.class_name = debouncedFilters.class_name;
    }

    if (debouncedFilters.branch) {
      params.branch = debouncedFilters.branch;
    }

    const data = await fetchStudents(params);

    setStudents(data.results);
    setCurrentPage(data.current_page);
    setTotalPages(data.total_pages);
    setCount(data.count);
  }, [debouncedFilters]);

  useEffect(() => {
    const loadBranches = async () => {
      setBranches(await fetchBranches());
    };
    loadBranches();
  }, []);

  useEffect(() => {
    loadStudents(currentPage);
  }, [currentPage, loadStudents]);

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

    loadStudents(currentPage);
  };

  const handleBulkPreview = async () => {
    if (!bulkFile) return alert("Select Excel file first");

    const fd = new FormData();
    fd.append("file", bulkFile);

    setBulkLoading(true);
    try {
      const data = await uploadStudentsExcel(fd, true);
      setPreviewData(data);
    } catch (err) {
      alert(err.message);
    }
    setBulkLoading(false);
  };

  const handleBulkCommit = async () => {
    if (!bulkFile) return alert("Select Excel file first");

    const fd = new FormData();
    fd.append("file", bulkFile);

    setBulkLoading(true);
    try {
      await uploadStudentsExcel(fd, false);
      alert("Students uploaded successfully");
      setPreviewData(null);
      setBulkFile(null);
      loadStudents(currentPage);
    } catch (err) {
      alert(err.message);
    }
    setBulkLoading(false);
  };

  const handleDownloadSample = async () => {
    const blob = await downloadSampleStudentsExcel();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample_students.xlsx";
    a.click();
  };

  return (
    <div className="container">
      <h1 className="title">Admin · Students</h1>

      {/* BULK UPLOAD */}
      <div className="box mb-5">
        <h2 className="subtitle">Bulk Upload Students</h2>

        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={(e) => setBulkFile(e.target.files[0])}
        />

        <div className="mt-3">
          <button
            className="button is-info mr-2"
            onClick={handleBulkPreview}
            disabled={bulkLoading}
          >
            Preview
          </button>

          <button
            className="button is-success mr-2"
            onClick={handleBulkCommit}
            disabled={bulkLoading}
          >
            Commit
          </button>

          <button
            className="button is-light"
            onClick={handleDownloadSample}
          >
            Download Sample
          </button>
        </div>

        {previewData && (
          <div className="mt-4">
            <p><strong>Success Count:</strong> {previewData.success_count}</p>

            <table className="table is-fullwidth is-striped is-small mt-2">
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Status</th>
                  <th>Error</th>
                </tr>
              </thead>
              <tbody>
                {previewData.results.map((r, i) => (
                  <tr key={i}>
                    <td>{r.row}</td>
                    <td>{r.status}</td>
                    <td>{r.error || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

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

      {/* FILTERS */}
      <div className="box mb-4">
        <h3 className="subtitle is-6">Filter Students</h3>

        <div className="columns is-multiline">

          <div className="column is-3">
            <input
              className="input"
              placeholder="Search by Name"
              value={filters.name}
              onChange={e =>
                setFilters({ ...filters, name: e.target.value })
              }
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Search by Roll No"
              value={filters.roll_no}
              onChange={e =>
                setFilters({ ...filters, roll_no: e.target.value })
              }
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Search by Class"
              value={filters.class_name}
              onChange={e =>
                setFilters({ ...filters, class_name: e.target.value })
              }
            />
          </div>

          <div className="column is-3">
            <div className="select is-fullwidth">
              <select
                value={filters.branch}
                onChange={e =>
                  setFilters({ ...filters, branch: e.target.value })
                }
              >
                <option value="">All Branches</option>
                {branches.map(b => (
                  <option key={b.id} value={b.id}>
                    {b.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

        </div>
      </div>

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
                    loadStudents(currentPage);
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

      {/* PAGINATION */}
      <div className="mt-4 is-flex is-justify-content-space-between is-align-items-center">
        <p>
          Total Students: <strong>{count}</strong>
        </p>

        <div>
          <button
            className="button mr-2"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(currentPage - 1)}
          >
            Previous
          </button>

          <span>
            Page {currentPage} of {totalPages}
          </span>

          <button
            className="button ml-2"
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(currentPage + 1)}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
