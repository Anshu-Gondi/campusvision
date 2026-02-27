import { useEffect, useState, useCallback } from "react";
import {
  fetchTeachers,
  createTeacher,
  updateTeacherImage,
  deleteTeacherImage,
  deleteTeacherFull,
  fetchBranches,
  uploadTeachersExcel,
  downloadSampleTeachersExcel
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

  const [bulkFile, setBulkFile] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [loadingBulk, setLoadingBulk] = useState(false);

  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [count, setCount] = useState(0);
  const pageSize = 10;

  const [filters, setFilters] = useState({
    name: "",
    employee_id: "",
    branch: "",
  });

  const [debouncedFilters, setDebouncedFilters] = useState(filters);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
      setCurrentPage(1); // reset page on filter change
    }, 400);

    return () => clearTimeout(timer);
  }, [filters]);

  useEffect(() => {
    const loadBranches = async () => {
      setBranches(await fetchBranches());
    };
    loadBranches();
  }, []);

  const loadTeachers = useCallback(async (page = 1) => {
    const params = {
      page,
      page_size: pageSize,
    };

    if (debouncedFilters.name) {
      params.name = debouncedFilters.name;
    }

    if (debouncedFilters.employee_id) {
      params.employee_id = debouncedFilters.employee_id;
    }

    if (debouncedFilters.branch) {
      params.branch = debouncedFilters.branch;
    }

    const data = await fetchTeachers(params);

    setTeachers(data.results);
    setCurrentPage(data.current_page);
    setTotalPages(data.total_pages);
    setCount(data.count);
  }, [debouncedFilters]);

  useEffect(() => {
    loadTeachers(currentPage);
  }, [currentPage, loadTeachers]);

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

    loadTeachers();
  };

  const uploadImage = async (id, file) => {
    const fd = new FormData();
    fd.append("image", file);
    await updateTeacherImage(id, fd);
    loadTeachers();
  };

  const handleBulkPreview = async () => {
    if (!bulkFile) return alert("Select Excel file first");

    const fd = new FormData();
    fd.append("file", bulkFile);

    try {
      setLoadingBulk(true);
      const data = await uploadTeachersExcel(fd, true);
      setPreviewData(data);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoadingBulk(false);
    }
  };

  const handleBulkCommit = async () => {
    if (!bulkFile) return alert("Select Excel file first");

    const fd = new FormData();
    fd.append("file", bulkFile);

    try {
      setLoadingBulk(true);
      await uploadTeachersExcel(fd, false);
      alert("Teachers uploaded successfully");
      setPreviewData(null);
      setBulkFile(null);
      loadTeachers();
    } catch (err) {
      alert(err.message);
    } finally {
      setLoadingBulk(false);
    }
  };

  const handleDownloadSample = async () => {
    const blob = await downloadSampleTeachersExcel();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample_teachers.xlsx";
    a.click();
  };

  return (
    <div className="container">
      <h1 className="title">Admin · Teachers</h1>

      {/* ================= BULK IMPORT ================= */}
      <div className="box">
        <h2 className="subtitle">Bulk Import Teachers (Excel)</h2>

        <div className="file mb-3">
          <label className="file-label">
            <input
              className="file-input"
              type="file"
              accept=".xls,.xlsx"
              onChange={e => setBulkFile(e.target.files[0])}
            />
            <span className="file-cta">
              <span className="file-label">
                Choose Excel File
              </span>
            </span>
          </label>
        </div>

        <div className="mt-3">
          <button
            className="button is-info mr-2"
            onClick={handleBulkPreview}
            disabled={loadingBulk}
          >
            Preview
          </button>

          <button
            className="button is-success mr-2"
            onClick={handleBulkCommit}
            disabled={loadingBulk}
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

        {previewData?.results && (
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
                {previewData.results?.map((r, i) => (
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

        {/* FILTERS */}
        <div className="box mb-4">
          <h3 className="subtitle is-6">Filter Teachers</h3>

          <div className="columns is-multiline">

            <div className="column is-4">
              <input
                className="input"
                placeholder="Search by Name"
                value={filters.name}
                onChange={e =>
                  setFilters({ ...filters, name: e.target.value })
                }
              />
            </div>

            <div className="column is-4">
              <input
                className="input"
                placeholder="Search by Employee ID"
                value={filters.employee_id}
                onChange={e =>
                  setFilters({ ...filters, employee_id: e.target.value })
                }
              />
            </div>

            <div className="column is-4">
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

        {/* PAGINATION */}
        <div className="mt-4 is-flex is-justify-content-space-between is-align-items-center">
          <p>
            Total Teachers: <strong>{count}</strong>
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
    </div>
  );
}
