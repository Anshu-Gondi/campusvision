import { useState, useEffect, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import {
  fetchTimetable,
  deleteTimetableEntry,
  uploadTimetable,
  updateTimetableEntry,
  downloadSampleTimetable,
  fetchTeachersList,
  fetchBranchesList,
} from "../../services/api";

export default function Timetable() {
  const queryClient = useQueryClient();

  // ---------------- FILTERS ----------------
  const [filters, setFilters] = useState({
    branch_id: "",
    class_name: "",
    section: "",
    teacher_id: "",
    teacher_search: "",
    day: "",
  });

  const [ordering, setOrdering] = useState("");
  const [page, setPage] = useState(1);
  const pageSize = 50;

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  // ---------------- DEBOUNCE ----------------
  const [debouncedTeacherSearch, setDebouncedTeacherSearch] = useState("");

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedTeacherSearch(filters.teacher_search);
    }, 400);
    return () => clearTimeout(handler);
  }, [filters.teacher_search]);

  // ---------------- DROPDOWNS ----------------
  const { data: teachersData = [] } = useQuery({
    queryKey: ["teachers-list"],
    queryFn: fetchTeachersList,
  });

  const { data: branchesData = [] } = useQuery({
    queryKey: ["branches-list"],
    queryFn: fetchBranchesList,
  });

  const teachers = useMemo(() => {
    return teachersData?.results || teachersData || [];
  }, [teachersData]);

  const branches = useMemo(() => {
    return branchesData?.results || branchesData || [];
  }, [branchesData]);

  // ---------------- TIMETABLE QUERY ----------------
  const timetableQuery = useQuery({
    queryKey: ["timetable", filters, ordering, page],
    queryFn: async () => {
      const params = {
        branch_id: filters.branch_id,
        teacher_id: filters.teacher_id,
        class_name: filters.class_name,
        section: filters.section,
        day: filters.day,
        ordering,
        page,
        page_size: pageSize,
      };

      const cleanParams = Object.fromEntries(
        Object.entries(params).filter(([, v]) => v !== "")
      );

      return fetchTimetable(cleanParams);
    },
    keepPreviousData: true,
  });

  const entries = timetableQuery.data?.results || [];
  const total = timetableQuery.data?.total || 0;

  // ---------------- OPTIMISTIC DELETE ----------------
  const deleteMutation = useMutation({
    mutationFn: deleteTimetableEntry,
    onMutate: async (id) => {
      setDeletingId(id);
      await queryClient.cancelQueries(["timetable"]);

      const previousData = queryClient.getQueryData([
        "timetable",
        filters,
        ordering,
        page,
      ]);

      queryClient.setQueryData(
        ["timetable", filters, ordering, page],
        (old) => ({
          ...old,
          results: old.results.filter((e) => e.id !== id),
        })
      );

      return { previousData };
    },
    onError: (err, id, context) => {
      queryClient.setQueryData(
        ["timetable", filters, ordering, page],
        context.previousData
      );
    },
    onSettled: () => {
      setDeletingId(null);
      queryClient.invalidateQueries(["timetable"]);
    },
  });

  // ---------------- UPDATE ----------------
  const updateMutation = useMutation({
    mutationFn: ({ id, data }) => updateTimetableEntry(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(["timetable"]);
    },
  });

  // ---------------- UPLOAD ----------------
  const uploadMutation = useMutation({
    mutationFn: ({ formData, preview }) =>
      uploadTimetable(formData, preview),
    onSuccess: () => {
      queryClient.invalidateQueries(["timetable"]);
    },
  });

  // ---------------- HANDLERS ----------------
  const handleFilterChange = (e) => {
    setFilters((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
    setPage(1);
  };

  const handleDelete = (id) => {
    if (!confirm("Delete this entry?")) return;
    deleteMutation.mutate(id);
  };

  const handleUpdate = (entry) => {
    const newSubject = prompt("Update subject:", entry.subject);
    if (!newSubject) return;

    updateMutation.mutate({
      id: entry.id,
      data: { subject: newSubject },
    });
  };

  const handlePreview = async () => {
    if (!selectedFile) return alert("Select file first");

    const formData = new FormData();
    formData.append("file", selectedFile);

    const preview = await uploadMutation.mutateAsync({
      formData,
      preview: true,
    });

    setPreviewData(preview);
  };

  const handleCommit = async () => {
    if (!selectedFile) return alert("Select file first");

    const formData = new FormData();
    formData.append("file", selectedFile);

    await uploadMutation.mutateAsync({
      formData,
      preview: false,
    });

    setPreviewData(null);
    setSelectedFile(null);
  };

  const handleDownloadSample = async () => {
    const blob = await downloadSampleTimetable();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample_timetable.xlsx";
    a.click();
  };

  const handleExport = async () => {
    const params = {
      branch_id: filters.branch_id,
      teacher_id: filters.teacher_id,
      class_name: filters.class_name,
      section: filters.section,
      day: filters.day,
      ordering,
      export: true,
    };

    const cleanParams = Object.fromEntries(
      Object.entries(params).filter(([, v]) => v !== "")
    );

    const blob = await fetchTimetable(cleanParams);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "filtered_timetable.xlsx";
    a.click();
  };

  // ---------------- TEACHER SEARCH ----------------
  const filteredTeachers = useMemo(() => {
    return teachers.filter((t) =>
      `${t.name} ${t.employee_id}`
        .toLowerCase()
        .includes(debouncedTeacherSearch.toLowerCase())
    );
  }, [teachers, debouncedTeacherSearch]);

  // ---------------- UI ----------------
  return (
    <div>
      <h1 className="title has-text-success">Timetable</h1>

      {/* Upload */}
      <div className="box">
        <input type="file" onChange={(e) => setSelectedFile(e.target.files[0])} />
        <div className="buttons mt-2">
          <button
            className="button is-info"
            disabled={uploadMutation.isPending}
            onClick={handlePreview}
          >
            {uploadMutation.isPending ? "Loading..." : "Preview"}
          </button>

          <button
            className="button is-success"
            disabled={uploadMutation.isPending}
            onClick={handleCommit}
          >
            {uploadMutation.isPending ? "Uploading..." : "Commit"}
          </button>

          <button className="button is-warning" onClick={handleDownloadSample}>
            Download Sample
          </button>
        </div>

        {previewData && (
          <div className="notification is-info mt-3">
            <p><strong>Valid Rows:</strong> {previewData.valid_count}</p>
            <p><strong>Errors:</strong> {previewData.error_count}</p>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="box">
        <div className="columns is-multiline">

          <div className="column is-3">
            <div className="select is-fullwidth">
              <select name="branch_id" value={filters.branch_id} onChange={handleFilterChange}>
                <option value="">All Branches</option>
                {branches.map(b => (
                  <option key={b.id} value={b.id}>{b.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="column is-2">
            <input className="input" name="class_name" placeholder="Class"
              value={filters.class_name} onChange={handleFilterChange} />
          </div>

          <div className="column is-2">
            <input className="input" name="section" placeholder="Section"
              value={filters.section} onChange={handleFilterChange} />
          </div>

          <div className="column is-3">
            <input className="input" name="teacher_search"
              placeholder="Search Teacher"
              value={filters.teacher_search}
              onChange={handleFilterChange} />
          </div>

          <div className="column is-2">
            <div className="select is-fullwidth">
              <select name="teacher_id" value={filters.teacher_id} onChange={handleFilterChange}>
                <option value="">All Teachers</option>
                {filteredTeachers.map(t => (
                  <option key={t.id} value={t.id}>
                    {t.name} ({t.employee_id})
                  </option>
                ))}
              </select>
            </div>
          </div>

        </div>
      </div>

      {/* Controls */}
      <div className="buttons mb-3">
        <button className="button is-small" onClick={() => setOrdering("day_of_week")}>Day ↑</button>
        <button className="button is-small" onClick={() => setOrdering("-day_of_week")}>Day ↓</button>
        <button className="button is-small" onClick={() => setOrdering("start_time")}>Time ↑</button>
        <button className="button is-small" onClick={() => setOrdering("-start_time")}>Time ↓</button>
        <button className="button is-primary" onClick={handleExport}>
          Export Filtered Data
        </button>
      </div>

      {/* Table */}
      {timetableQuery.isLoading ? (
        <p>Loading timetable...</p>
      ) : (
        <>
          <table className="table is-fullwidth is-striped is-dark">
            <thead>
              <tr>
                <th>Branch</th>
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
                  <td>{e.branch}</td>
                  <td>{e.day_of_week}</td>
                  <td>{e.class_name}</td>
                  <td>{e.section}</td>
                  <td>{e.subject}</td>
                  <td>{e.teacher_name}</td>
                  <td>{e.start_time} - {e.end_time}</td>
                  <td>
                    <div className="buttons are-small">
                      <button className="button is-link"
                        onClick={() => handleUpdate(e)}>
                        Edit
                      </button>

                      <button
                        className="button is-danger"
                        disabled={deletingId === e.id}
                        onClick={() => handleDelete(e.id)}
                      >
                        {deletingId === e.id ? "Deleting..." : "Delete"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <nav className="pagination is-centered mt-4">
            <button
              className="pagination-previous button"
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
            >
              Previous
            </button>

            <button
              className="pagination-next button"
              disabled={page * pageSize >= total}
              onClick={() => setPage(p => p + 1)}
            >
              Next
            </button>

            <span className="pagination-link">
              Page {page} of {Math.ceil(total / pageSize) || 1}
            </span>
          </nav>
        </>
      )}
    </div>
  );
}