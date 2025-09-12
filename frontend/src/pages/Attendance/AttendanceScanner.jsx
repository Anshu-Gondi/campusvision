import { useState, useEffect, useCallback } from "react";
import { QrReader } from "react-qr-reader";
import { QRCodeCanvas } from "qrcode.react";
import { createQrSession } from "../../services/api";
import { useNavigate } from "react-router-dom";

export default function AttendanceScanner() {
  const [qrValue, setQrValue] = useState("");
  const [prevQrValue, setPrevQrValue] = useState(""); // hold previous QR for grace overlap
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const generateQr = useCallback(async () => {
    try {
      const data = await createQrSession();
      if (data.code) {
        const reactFormUrl = `${window.location.origin}/attendance/form`;

        // move current to previous before updating
        setPrevQrValue(qrValue);
        setQrValue(
          `${import.meta.env.VITE_API_KEY}/qr/validate/${data.code}/?redirect=${encodeURIComponent(
            reactFormUrl
          )}`
        );
      }
    } catch (err) {
      console.error("QR session error:", err);
    }
  }, [qrValue]);

  useEffect(() => {
    generateQr(); // run once

    const interval = setInterval(() => {
      generateQr();
    }, 20000);

    return () => clearInterval(interval);
  }, [generateQr]);


  const handleQrScan = (result) => {
    if (result?.text) {
      const code = result.text.trim();

      // ignore scanning the same QR as previous (grace)
      if (code === qrValue || code === prevQrValue) return;

      setMessage(`Scanned QR: ${code}`);
      navigate("/attendance/form?code=" + encodeURIComponent(code));
    }
  };

  return (
    <div className="container has-text-centered">
      <h1 className="title has-text-white">Scan QR Code For Attendance Marking Progress Starting</h1>

      {qrValue && (
        <div>
          <p className="subtitle">📱 Scan this QR code to start</p>
          <QRCodeCanvas
            value={qrValue}
            size={256}
            bgColor="#ffffff"
            fgColor="#008000"
            style={{ margin: "20px auto", boxShadow: "0 0 15px yellow" }}
          />
        </div>
      )}

      <div style={{ width: 300, height: 300, margin: "auto" }}>
        <QrReader
          onResult={handleQrScan}
          scanDelay={300}
          constraints={{ facingMode: "environment" }}
          onError={() => setMessage("QR scanner error")}
          videoStyle={{ width: "100%", height: "100%" }}
        />
      </div>

      {message && <p className="has-text-info mt-3">{message}</p>}
    </div>
  );
}
