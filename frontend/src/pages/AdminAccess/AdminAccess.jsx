import React, { useEffect, useState } from "react";
import { adminLogin } from "../../services/api";

export default function AdminAccess() {
  const [combo, setCombo] = useState([]);
  const [display, setDisplay] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const handler = (e) => {
      let key = null;

      // Handle arrows
      if (e.key.startsWith("Arrow")) {
        key = {
          ArrowUp: "↑",
          ArrowDown: "↓",
          ArrowLeft: "←",
          ArrowRight: "→",
        }[e.key];
      }
      // Handle letters & numbers
      else if (/^[a-zA-Z0-9]$/.test(e.key)) {
        key = e.key.toUpperCase();
      }

      if (!key) return;

      setCombo(prev => [...prev, key]);
      setDisplay(prev => prev + "• ");
      e.preventDefault();
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const submit = async () => {
    const finalCombo = combo.join("");
    const res = await adminLogin(finalCombo);

    if (res.success) {
      localStorage.setItem("admin_token", res.token);
      window.location.href = "/admin";
    } else {
      setMessage("❌ Invalid combo");
      reset();
    }
  };

  const reset = () => {
    setCombo([]);
    setDisplay("");
    setMessage("");
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>SECURE ADMIN ACCESS</h2>
      <div style={styles.comboBox}>
        {display || "Press secret keys..."}
      </div>
      <div style={{ marginTop: 20 }}>
        <button onClick={submit} style={styles.button}>UNLOCK</button>
        <button onClick={reset} style={styles.reset}>RESET</button>
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
  comboBox: {
    width: 320,
    minHeight: 50,
    border: "1px solid #0f0",
    padding: 12,
    textAlign: "center",
    background: "#050505",
    boxShadow: "0 0 15px #0f0",
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
