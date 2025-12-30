const API_BASE = import.meta.env.VITE_API_KEY;

// ================= ADMIN STUDENTS =================

export const fetchStudents = async () => {
  const res = await fetch(`${API_BASE}/admin/students/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
  });
  return res.json();
};

export const createStudent = async (formData) => {
  const res = await fetch(`${API_BASE}/admin/students/`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(formData),
  });
  return res.json();
};

export const updateStudentImage = async (studentId, formData) => {
  const res = await fetch(
    `${API_BASE}/admin/students/${studentId}/image/`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
      body: formData,
    }
  );
  return res.json();
};

export const deleteStudentImage = async (studentId) => {
  const res = await fetch(
    `${API_BASE}/admin/students/${studentId}/image/delete/`,
    {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }
  );
  return res.json();
};

export const deleteStudentFull = async (studentId) => {
  const res = await fetch(
    `${API_BASE}/admin/students/${studentId}/`,
    {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }
  );
  return res.json();
};

// ================= ADMIN TEACHERS =================

export const fetchTeachers = async () => {
  const res = await fetch(`${API_BASE}/admin/teachers/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
  });
  return res.json();
};

export const createTeacher = async (formData) => {
  const res = await fetch(`${API_BASE}/admin/teachers/`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
    },
    body: JSON.stringify(formData),
  });
  return res.json();
};

export const updateTeacherImage = async (teacherId, formData) => {
  const res = await fetch(
    `${API_BASE}/admin/teachers/${teacherId}/image/`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
      body: JSON.stringify(formData),
    }
  );
  return res.json();
};

export const deleteTeacherImage = async (teacherId) => {
  const res = await fetch(
    `${API_BASE}/admin/teachers/${teacherId}/image/delete/`,
    {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }
  );
  return res.json();
};

export const deleteTeacherFull = async (teacherId) => {
  const res = await fetch(
    `${API_BASE}/admin/teachers/${teacherId}/`,
    {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }
  );
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
