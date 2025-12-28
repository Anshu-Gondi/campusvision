import { useEffect, useMemo, useState } from "react";
import MapPicker from "./Mappicker";
import {
  fetchBranches,
  createBranch,
  updateBranch,
  deleteBranch,
} from "../../services/api";

/**
 * Renovated Admin Branches Page
 * - Organization name shown once (admin is org-bound)
 * - Branch list grouped under the organization
 * - Full CRUD
 * - Structured location (address, city, state, pincode, lat/long)
 */
export default function Branches() {
  const [branches, setBranches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mapLink, setMapLink] = useState("");

  const [form, setForm] = useState({
    name: "",
    address: "",
    city: "",
    state: "",
    pincode: "",
    latitude: "",
    longitude: "",
    attendance_radius: 75,
  });

  const [editingId, setEditingId] = useState(null);

  useEffect(() => {
    loadBranches();
  }, []);

  const loadBranches = async () => {
    setLoading(true);
    const data = await fetchBranches();
    setBranches(data);
    setLoading(false);
  };

  const resetForm = () => {
    setEditingId(null);
    setForm({
      name: "",
      address: "",
      city: "",
      state: "",
      pincode: "",
      latitude: "",
      longitude: "",
    });
  };

  const submit = async () => {
    if (!form.name.trim()) {
      alert("Branch name is required");
      return;
    }

    const payload = {
      ...form,
      latitude: form.latitude || null,
      longitude: form.longitude || null,
    };

    if (editingId) {
      await updateBranch(editingId, payload);
    } else {
      await createBranch(payload);
    }

    resetForm();
    loadBranches();
  };

  const editBranch = (b) => {
    setEditingId(b.id);
    setForm({
      name: b.name || "",
      address: b.address || "",
      city: b.city || "",
      state: b.state || "",
      pincode: b.pincode || "",
      latitude: b.latitude || "",
      longitude: b.longitude || "",
    });
  };

  const removeBranch = async (id) => {
    if (!window.confirm("Delete this branch?")) return;
    await deleteBranch(id);
    loadBranches();
  };

  const organizationName = useMemo(() => {
    if (!branches.length) return "";
    return branches[0].organization_name;
  }, [branches]);

  const extractLatLng = (url) => {
    const atMatch = url.match(/@(-?\d+\.\d+),(-?\d+\.\d+)/);
    if (atMatch) return { lat: atMatch[1], lng: atMatch[2] };

    const qMatch = url.match(/q=(-?\d+\.\d+),(-?\d+\.\d+)/);
    if (qMatch) return { lat: qMatch[1], lng: qMatch[2] };

    return null;
  };

  const handleMapLink = async (url) => {
    setMapLink(url);
    const coords = extractLatLng(url);
    if (!coords) return;

    await reverseGeocodeFromParent(coords.lat, coords.lng);
  };

  const getCurrentLocation = () => {
    if (!navigator.geolocation) return;

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        reverseGeocodeFromParent(
          pos.coords.latitude.toFixed(6),
          pos.coords.longitude.toFixed(6)
        );
      },
      () => alert("Location permission denied")
    );
  };

  const reverseGeocodeFromParent = async (lat, lng) => {
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
      );
      const data = await res.json();
      const addr = data.address || {};

      setForm((prev) => ({
        ...prev,
        latitude: lat,
        longitude: lng,
        address: addr.road || addr.suburb || addr.neighbourhood || prev.address,
        city: addr.city || addr.town || addr.village || prev.city,
        state: addr.state || prev.state,
        pincode: addr.postcode || prev.pincode,
      }));
    } catch (e) {
      console.error("Reverse geocode failed", e);
    }
  };

  return (
    <div className="p-4">
      <h1 className="title is-4">Organization</h1>
      <div className="box mb-5">
        <strong>{organizationName || "Your Organization"}</strong>
      </div>

      {/* Branch Form */}
      <div className="box mb-6">
        <h2 className="subtitle is-6">
          {editingId ? "Edit Branch" : "Create Branch"}
        </h2>

        <div className="columns is-multiline">
          <div className="column is-12">
            <button
              className="button is-link mb-2"
              onClick={getCurrentLocation}
            >
              📍 Use my current location
            </button>

            <input
              className="input"
              placeholder="Paste Google Maps link"
              value={mapLink}
              onChange={(e) => handleMapLink(e.target.value)}
            />
          </div>

          {/* Map Picker */}
          <div className="column is-12">
            <div className="box">
              <p className="mb-2">
                <strong>Select location on map</strong>
              </p>
              <MapPicker
                latitude={form.latitude}
                longitude={form.longitude}
                radius={form.attendance_radius}
                onChange={(data) =>
                  setForm({
                    ...form,
                    latitude: data.latitude,
                    longitude: data.longitude,
                    address: data.address || form.address,
                    city: data.city || form.city,
                    state: data.state || form.state,
                    pincode: data.pincode || form.pincode,
                  })
                }
              />
            </div>
          </div>

          <div className="column is-4">
            <input
              className="input"
              placeholder="Branch Name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>

          <div className="column is-8">
            <input
              className="input"
              placeholder="Address"
              value={form.address}
              onChange={(e) => setForm({ ...form, address: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="City"
              value={form.city}
              onChange={(e) => setForm({ ...form, city: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="State"
              value={form.state}
              onChange={(e) => setForm({ ...form, state: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Pincode"
              value={form.pincode}
              onChange={(e) => setForm({ ...form, pincode: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Latitude"
              value={form.latitude}
              onChange={(e) => setForm({ ...form, latitude: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              type="number"
              min="10"
              placeholder="Attendance Radius (meters)"
              value={form.attendance_radius}
              onChange={(e) =>
                setForm({ ...form, attendance_radius: e.target.value })
              }
            />
          </div>

          <div className="column is-3">
            <input
              className="input"
              placeholder="Longitude"
              value={form.longitude}
              onChange={(e) => setForm({ ...form, longitude: e.target.value })}
            />
          </div>

          <div className="column is-3">
            <button className="button is-success is-fullwidth" onClick={submit}>
              {editingId ? "Update Branch" : "Create Branch"}
            </button>
          </div>

          {editingId && (
            <div className="column is-3">
              <button
                className="button is-light is-fullwidth"
                onClick={resetForm}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Branch List */}
      <div className="box">
        <h2 className="subtitle is-6">Branches</h2>

        {loading ? (
          <p>Loading...</p>
        ) : (
          <table className="table is-fullwidth is-striped">
            <thead>
              <tr>
                <th>Name</th>
                <th>City</th>
                <th>State</th>
                <th>Location</th>
                <th width="160">Actions</th>
              </tr>
            </thead>
            <tbody>
              {branches.map((b) => (
                <tr key={b.id}>
                  <td>{b.name}</td>
                  <td>{b.city || "—"}</td>
                  <td>{b.state || "—"}</td>
                  <td>{b.address || "—"}</td>
                  <td>
                    <button
                      className="button is-small is-info mr-2"
                      onClick={() => editBranch(b)}
                    >
                      Edit
                    </button>
                    <button
                      className="button is-small is-danger"
                      onClick={() => removeBranch(b.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
