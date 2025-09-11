import { useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { verifyAttendance } from "../../services/api";

export default function AttendanceCamera() {
  const location = useLocation();
  const navigate = useNavigate();
  const { form, qrCode } = location.state || {};
  const [message, setMessage] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const requestCameraAndCapture = async () => {
    setMessage("");

    // 1️⃣ Ask user for consent
    const userConsent = window.confirm(
      "📸 This website requires camera access to capture your face for attendance. Do you allow?"
    );
    if (!userConsent) {
      setMessage("⚠️ Camera access denied by user");
      return;
    }

    let stream;
    try {
      // 2️⃣ Request front camera
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" }, // front camera
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera error:", err);

      // 🔎 Detect specific issues
      if (window.location.protocol !== "https:") {
        setMessage(
          "⚠️ Camera blocked: Browsers require HTTPS (or localhost) to access the camera."
        );
      } else if (err.name === "NotAllowedError") {
        setMessage("⚠️ Camera permission denied. Please allow camera access.");
      } else if (err.name === "NotFoundError") {
        setMessage("⚠️ No camera found on this device.");
      } else {
        setMessage("⚠️ Unable to access camera. Please try again.");
      }
      return;
    }

    // 3️⃣ Capture frame
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 4️⃣ Convert to blob and submit
    canvas.toBlob(
      async (blob) => {
        const formData = new FormData();
        formData.append("type", form.type);
        formData.append("id", form.id);
        formData.append("image", blob, "capture.jpg");
        formData.append("qr_code", qrCode);

        try {
          const data = await verifyAttendance(formData);
          setMessage(
            data?.error ? `❌ ${data.error}` : "✅ Attendance submitted!"
          );
        } catch {
          setMessage("❌ Network error");
        } finally {
          // 5️⃣ Stop camera to force next permission prompt
          stream.getTracks().forEach((track) => track.stop());
          setTimeout(() => navigate("/attendance/scan"), 5000);
        }
      },
      "image/jpeg"
    );
  };

  return (
    <div className="container has-text-centered">
      <h1 className="title has-text-primary">Face Capture</h1>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "100%", maxWidth: 400 }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <button
        className="button is-success mt-3"
        onClick={requestCameraAndCapture}
      >
        Capture & Submit
      </button>

      {message && (
        <p
          className={`mt-3 ${
            message.startsWith("✅")
              ? "has-text-success"
              : "has-text-danger has-text-weight-bold"
          }`}
        >
          {message}
        </p>
      )}
    </div>
  );
}
