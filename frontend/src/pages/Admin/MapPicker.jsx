import { MapContainer, TileLayer, Marker, useMapEvents, Circle } from "react-leaflet";
import { useState } from "react";
import L from "leaflet";

// Fix default marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl:
        "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    iconUrl:
        "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    shadowUrl:
        "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function ClickHandler({ onPick }) {
    useMapEvents({
        click(e) {
            onPick(e.latlng.lat.toFixed(6), e.latlng.lng.toFixed(6));
        },
    });
    return null;
}

const reverseGeocode = async (lat, lng) => {
    try {
        const res = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
        );
        const data = await res.json();

        const address = data.address || {};

        return {
            address_line:
                address.road ||
                address.suburb ||
                address.neighbourhood ||
                address.hamlet ||
                "",
            city:
                address.city ||
                address.town ||
                address.village ||
                address.county ||
                "",
            state: address.state || "",
            pincode: address.postcode || "",
        };
    } catch (err) {
        console.error("Reverse geocoding failed", err);
        return {
            address_line: "",
            city: "",
            state: "",
            pincode: "",
        };
    }
};

export default function MapPicker({ latitude, longitude, onChange, radius }) {
    const defaultCenter = latitude && longitude
        ? [latitude, longitude]
        : [20.5937, 78.9629]; // India default

    const [position, setPosition] = useState(
        latitude && longitude ? [latitude, longitude] : null
    );

    const handlePick = async (lat, lng) => {
        setPosition([lat, lng]);

        const location = await reverseGeocode(lat, lng);

        onChange({
            latitude: lat,
            longitude: lng,
            address: location.address_line,
            city: location.city,
            state: location.state,
            pincode: location.pincode,
        });
    };

    return (
        <MapContainer
            center={defaultCenter}
            zoom={latitude ? 15 : 5}
            style={{ height: "300px", width: "100%" }}
        >
            <TileLayer
                attribution="© OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <ClickHandler onPick={handlePick} />
            {position && (
                <>
                    <Marker position={position} />
                    <Circle
                        center={position}
                        radius={radius || 75}
                        pathOptions={{ fillOpacity: 0.2 }}
                    />
                </>
            )}
        </MapContainer>
    );
}
