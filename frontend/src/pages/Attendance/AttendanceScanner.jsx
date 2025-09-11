import { useState, useEffect } from "react";
import { QrReader } from "react-qr-reader";
import { QRCodeCanvas } from "qrcode.react";
import { createQrSession } from "../../services/api";
import { useNavigate } from "react-router-dom";

export default function AttendanceScanner() {
  const [qrValue, setQrValue] = useState("");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const generateQr = async () => {
    const data = await createQrSession();
    if (data.code) {
      const reactFormUrl = `${window.location.origin}/attendance/form`;
      setQrValue(`${import.meta.env.VITE_API_KEY}/qr/validate/${data.code}/?redirect=${encodeURIComponent(reactFormUrl)}`);
    }
  };

  useEffect(() => {
    generateQr();
  }, []);

  const handleQrScan = (result) => {
    if (result?.text) {
      // Normally Django redirects, but you can also handle local scan for testing
      const code = result.text.trim();
      setMessage(`Scanned QR: ${code}`);
      navigate("/attendance/form?code=" + encodeURIComponent(code));
    }
  };

  return (
    <div className="container has-text-centered">
      <h1 className="title has-text-primary">Scan QR Code</h1>

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
