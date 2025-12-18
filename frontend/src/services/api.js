const API_BASE = import.meta.env.VITE_API_KEY;

// ---------------- STUDENTS ----------------
export const fetchStudents = async () => {
  const res = await fetch(`${API_BASE}/students/`);
  return res.json();
};

export const createStudent = async (formData) => {
  const res = await fetch(`${API_BASE}/register/student/`, {
    method: "POST",
    body: formData,
  });
  return res.json();
};

export const updateStudentImage = async (studentId, formData) => {
  const res = await fetch(`${API_BASE}/upload/student/${studentId}/`, {
    method: "POST",
    body: formData,
  });
  return res.json();
};

// 🟢 Old (image-only delete)
export const deleteStudent = async (rollNo) => {
  const res = await fetch(`${API_BASE}/delete/student/image/${rollNo}/`, {
    method: "DELETE",
  });
  return res.json();
};

// 🆕 New (full delete: student + attendance + image)
export const deleteStudentFull = async (rollNo) => {
  const res = await fetch(`${API_BASE}/delete/student/${rollNo}/`, {
    method: "DELETE",
  });
  return res.json();
};

// ---------------- TEACHERS ----------------
export const fetchTeachers = async () => {
  const res = await fetch(`${API_BASE}/teachers/`);
  return res.json();
};

export const createTeacher = async (formData) => {
  const res = await fetch(`${API_BASE}/register/teacher/`, {
    method: "POST",
    body: formData,
  });
  return res.json();
};

export const updateTeacherImage = async (teacherId, formData) => {
  const res = await fetch(`${API_BASE}/upload/teacher/${teacherId}/`, {
    method: "POST",
    body: formData,
  });
  return res.json();
};

// 🟢 Old (image-only delete)
export const deleteTeacher = async (employeeId) => {
  const res = await fetch(`${API_BASE}/delete/teacher/image/${employeeId}/`, {
    method: "DELETE",
  });
  return res.json();
};

// 🆕 New (full delete: teacher + attendance + image)
export const deleteTeacherFull = async (employeeId) => {
  const res = await fetch(`${API_BASE}/delete/teacher/${employeeId}/`, {
    method: "DELETE",
  });
  return res.json();
};

// ---------------- ATTENDANCE ----------------
export const verifyAttendance = async (formData) => {
  const res = await fetch(`${API_BASE}/verify/`, {
    method: "POST",
    body: formData,
  });
  return res.json();
};

// ---------------- QR SESSION ----------------
export const createQrSession = async () => {
  const res = await fetch(`${API_BASE}/qr/create/`, { method: "POST" });
  return res.json();
};

// ✅ FIXED: Now takes only code, not full URL
export const validateQrSession = async (code) => {
  const res = await fetch(`${API_BASE}/qr/validate/${code}/`);
  return res.json();
};


// ---------------- ADMIN AUTHENTICATION ----------------
export const adminLogin = async ({organization, combo}) => {
  const res = await fetch(`${API_BASE}/admin/login/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ organization, combo }),
  });
  return res.json();
};

// ---------------- Organizations Apis ----------------
export const getOrganization = async () => {
  const res = await fetch(`${API_BASE}/admin/organization/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
  });
  return res.json();
};

export const updateOrganization = async (data) => {
  await fetch(`${API_BASE}/admin/organization/update/`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(data),
  });
};

export const rotateAdminCombo = async (data) => {
  await fetch(`${API_BASE}/admin/organization/rotate-combo/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(data),
  });
};

export const createOrganization = async (data) => {
  return fetch(`${API_BASE}/organizations/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(data),
  });
};

export const fetchBranches = async () => {
  const res = await fetch(`${API_BASE}/admin/branches/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
  });
  return res.json();
};

export const createBranch = async (data) => {
  return fetch(`${API_BASE}/admin/branches/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(data),
  });
};

export const updateBranch = async (id, data) => {
  return fetch(`${API_BASE}/admin/branches/${id}/`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(data),
  });
};

export const deleteBranch = async (id) => {
  return fetch(`${API_BASE}/admin/branches/${id}/`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
  });
};
