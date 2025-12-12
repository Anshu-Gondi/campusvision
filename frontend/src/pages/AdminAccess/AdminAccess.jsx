import React, { useState } from "react";
import { adminLogin } from "../../services/api";

export default function AdminAccess() {
  const [combo, setCombo] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async () => {
    const res = await adminLogin(combo);

    if (res.success) {
      localStorage.setItem("admin_token", res.token);
      window.location.href = "/admin-dashboard";
    } else {
      setMessage("Invalid combo");
    }
  };

  return (
    <div style={{
      height: "100vh",
      background: "black",
      color: "#0f0",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      alignItems: "center"
    }}>
      <h2 style={{ marginBottom: 20 }}>Secure Admin Access</h2>

      <input
        value={combo}
        onChange={(e) => setCombo(e.target.value)}
        style={{
          width: "300px",
          padding: "12px",
          background: "#111",
          border: "1px solid #0f0",
          color: "#0f0",
          outline: "none"
        }}
        placeholder="Enter secret combo e.g. C 3 C3↑"
      />

      <button
        onClick={handleSubmit}
        style={{
          marginTop: 20,
          padding: "10px 20px",
          background: "#0f0",
          color: "black",
          cursor: "pointer"
        }}
      >
        Unlock
      </button>

      {message && <div style={{ marginTop: 15, color: "red" }}>{message}</div>}
    </div>
  );
}
