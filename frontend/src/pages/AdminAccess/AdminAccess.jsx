import React, { useEffect, useState, useRef } from "react";
import { adminLogin } from "../../services/api";

export default function AdminAccess() {
  const [organization, setOrganization] = useState("");
  const [combo, setCombo] = useState([]);
  const [display, setDisplay] = useState("");
  const [message, setMessage] = useState("");
  const comboRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      let key = null;

      if (e.key.startsWith("Arrow")) {
        key = {
          ArrowUp: "↑",
          ArrowDown: "↓",
          ArrowLeft: "←",
          ArrowRight: "→",
        }[e.key];
      } else if (/^[a-zA-Z0-9]$/.test(e.key)) {
        key = e.key.toUpperCase();
      }

      if (!key) return;

      setCombo((prev) => [...prev, key]);
      setDisplay((prev) => prev + "• ");
      e.preventDefault();
    };

    const comboEl = comboRef.current;
    if (comboEl) comboEl.addEventListener("keydown", handler);

    return () => {
      if (comboEl) comboEl.removeEventListener("keydown", handler);
    };
  }, []);

  const submit = async () => {
    setMessage("");

    if (!organization.trim()) {
      setMessage("⚠ Organization name required");
      return;
    }

    if (combo.length === 0) {
      setMessage("⚠ Enter admin key combo");
      return;
    }

    try {
      const finalCombo = combo.join("");

      const res = await adminLogin({
        organization: organization.trim(),
        combo: finalCombo,
      });

      // 🔴 Organization not found
      if (res.error) {
        setMessage(`❌ ${res.error}`);
        return;
      }

      // 🔴 Wrong combo
      if (!res.success) {
        setMessage(`❌ ${res.message || "Invalid credentials"}`);
        reset(false);
        return;
      }

      // ✅ Success
      localStorage.setItem("admin_token", res.token);
      localStorage.setItem("org_name", res.organization);
      window.location.href = "/admin";

    } catch (err) {
      console.error(err);
      setMessage("❌ Server error. Please try again.");
    }
  };

  const reset = (clearOrg = true) => {
    setCombo([]);
    setDisplay("");
    setMessage("");
    if (clearOrg) setOrganization("");
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>SECURE ADMIN ACCESS</h2>

      {/* Organization Input */}
      <input
        style={styles.input}
        placeholder="Organization name"
        value={organization}
        onChange={(e) => setOrganization(e.target.value)}
      />

      {/* Combo Display (focusable) */}
      <div
        style={styles.comboBox}
        tabIndex={0}
        ref={comboRef}
      >
        {display || "Click here and press secret keys..."}
      </div>

      <div style={{ marginTop: 20 }}>
        <button onClick={submit} style={styles.button}>
          UNLOCK
        </button>
        <button onClick={() => reset()} style={styles.reset}>
          RESET
        </button>
      </div>

      {message && <p style={styles.error}>{message}</p>}
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    background: "black",
    color: "#0f0",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "monospace",
  },
  title: {
    marginBottom: 25,
    letterSpacing: 2,
  },
  input: {
    width: 320,
    padding: 10,
    marginBottom: 15,
    background: "#000",
    color: "#0f0",
    border: "1px solid #0f0",
    outline: "none",
    textAlign: "center",
    fontFamily: "monospace",
  },
  comboBox: {
    width: 320,
    minHeight: 50,
    border: "1px solid #0f0",
    padding: 12,
    textAlign: "center",
    background: "#050505",
    boxShadow: "0 0 15px #0f0",
    cursor: "text",
  },
  button: {
    padding: "10px 20px",
    background: "#0f0",
    border: "none",
    cursor: "pointer",
    color: "white",
    marginRight: 10,
  },
  reset: {
    padding: "10px 20px",
    background: "#111",
    color: "#0f0",
    border: "1px solid #0f0",
    cursor: "pointer",
  },
  error: {
    marginTop: 15,
    color: "red",
  },
};
