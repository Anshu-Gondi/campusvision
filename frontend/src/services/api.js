const API_BASE = import.meta.env.VITE_API_KEY;

// ================= CORE REQUEST WRAPPER =================

const request = async (url, options = {}) => {
  const token = localStorage.getItem("admin_token");

  const config = {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  };

  const res = await fetch(url, config);

  if (!res.ok) {
    let errorMessage = "Request failed";
    try {
      const errorData = await res.json();
      errorMessage = errorData?.error || errorMessage;
    } catch {
      /* ignore */
    }
    throw new Error(errorMessage);
  }

  if (res.status === 204) return null;

  // 👇 KEY ADDITION
  const contentType = res.headers.get("content-type");

  if (contentType?.includes("application/json")) {
    return res.json();
  }

  if (contentType?.includes("application/octet-stream") ||
      contentType?.includes("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")) {
    return res.blob();
  }

  // fallback
  return res.text();
};

// ================= ADMIN STUDENTS =================

export const fetchStudents = (params = {}) => {
  const queryString = new URLSearchParams(params).toString();
  const url = queryString
    ? `${API_BASE}/admin/students/?${queryString}`
    : `${API_BASE}/admin/students/`;

  return request(url);
};

export const createStudent = (data) =>
  request(`${API_BASE}/admin/students/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const updateStudentImage = (studentId, formData) =>
  request(`${API_BASE}/admin/students/${studentId}/image/`, {
    method: "POST",
    body: formData,
  });

export const deleteStudentImage = (studentId) =>
  request(`${API_BASE}/admin/students/${studentId}/image/delete/`, {
    method: "DELETE",
  });

export const deleteStudentFull = (studentId) =>
  request(`${API_BASE}/admin/students/${studentId}/`, {
    method: "DELETE",
  });

// ================= ADMIN TEACHERS =================

export const fetchTeachers = (params = {}) => {
  const queryString = new URLSearchParams(params).toString();
  const url = queryString
    ? `${API_BASE}/admin/teachers/?${queryString}`
    : `${API_BASE}/admin/teachers/`;

  return request(url);
};

export const createTeacher = (data) =>
  request(`${API_BASE}/admin/teachers/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const updateTeacherImage = (teacherId, formData) =>
  request(`${API_BASE}/admin/teachers/${teacherId}/image/`, {
    method: "POST",
    body: formData,
  });

export const deleteTeacherImage = (teacherId) =>
  request(`${API_BASE}/admin/teachers/${teacherId}/image/delete/`, {
    method: "DELETE",
  });

export const deleteTeacherFull = (teacherId) =>
  request(`${API_BASE}/admin/teachers/${teacherId}/`, {
    method: "DELETE",
  });

// ================= BULK USER UPLOAD =================

// Upload Teachers Excel (preview or commit)
export const uploadTeachersExcel = (formData, preview = true) =>
  request(`${API_BASE}/admin/user-upload/teachers/?preview=${preview}`, {
    method: "POST",
    body: formData,
  });

// Upload Students Excel (preview or commit)
export const uploadStudentsExcel = (formData, preview = true) =>
  request(`${API_BASE}/admin/user-upload/students/?preview=${preview}`, {
    method: "POST",
    body: formData,
  });

// Download Sample Teachers Excel
export const downloadSampleTeachersExcel = () =>
  request(`${API_BASE}/admin/user-upload/download-sample-teachers/`);

// Download Sample Students Excel
export const downloadSampleStudentsExcel = () =>
  request(`${API_BASE}/admin/user-upload/download-sample-students/`);

// ================= ATTENDANCE =================

export const verifyAttendance = (formData) =>
  request(`${API_BASE}/verify/`, {
    method: "POST",
    body: formData,
  });

// ================= QR SESSION =================

export const createQrSession = () =>
  request(`${API_BASE}/qr/create/`, {
    method: "POST",
  });

export const validateQrSession = (code) =>
  request(`${API_BASE}/qr/validate/${code}/`);

// ================= ADMIN AUTH =================

export const adminLogin = ({ organization, combo }) =>
  request(`${API_BASE}/admin/login/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ organization, combo }),
  });

// ================= ORGANIZATION =================

export const getOrganization = () => request(`${API_BASE}/admin/organization/`);

export const updateOrganization = (data) =>
  request(`${API_BASE}/admin/organization/update/`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const rotateAdminCombo = (data) =>
  request(`${API_BASE}/admin/organization/rotate-combo/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const createOrganization = (data) =>
  request(`${API_BASE}/organizations/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

// ================= BRANCHES =================

export const fetchBranches = () => request(`${API_BASE}/admin/branches/`);

export const createBranch = (data) =>
  request(`${API_BASE}/admin/branches/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const updateBranch = (id, data) =>
  request(`${API_BASE}/admin/branches/${id}/`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const deleteBranch = (id) =>
  request(`${API_BASE}/admin/branches/${id}/`, {
    method: "DELETE",
  });

// ================= TIMETABLE =================

// List with optional query params
export const fetchTimetable = (params = {}) => {
  // Convert params object to query string
  const queryString = new URLSearchParams(params).toString();
  const url = queryString
    ? `${API_BASE}/admin/timetable/?${queryString}`
    : `${API_BASE}/admin/timetable/`;
  return request(url);
};

// Upload Excel (preview or commit)
export const uploadTimetable = (formData, preview = true) =>
  request(`${API_BASE}/admin/timetable/upload/?preview=${preview}`, {
    method: "POST",
    body: formData,
  });

// Download sample
export const downloadSampleTimetable = () =>
  request(`${API_BASE}/admin/timetable/download-sample-time-table`);

// Update entry
export const updateTimetableEntry = (id, data) =>
  request(`${API_BASE}/admin/timetable/${id}/update/`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

// Delete entry
export const deleteTimetableEntry = (id) =>
  request(`${API_BASE}/admin/timetable/${id}/delete/`, {
    method: "DELETE",
  });

// ================= SCHEDULER =================

export const generateSubstitution = (data) =>
  request(`${API_BASE}/scheduler/substitution/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
