# rust_backend

Location: Backend/attendance_backend/rust_extensions/rust_backend

A Rust-based extension crate that provides ultra-fast face recognition, anti-spoofing helper utilities and CCTV tracking primitives exposed to Python via pyo3/maturin. The crate is compiled as a cdylib and packaged as a Python extension module named `rust_backend`.

This README documents what the crate does, what it contains, how to build it, and how to use / integrate it from Python.

---

## High level overview

- Purpose: performance-critical functionality for the Face Recognition Attendance System implemented in Rust and exposed to Python for low-latency face embedding extraction, approximate nearest neighbor search (ANN) for matching, preprocessing and CCTV-specific tracking/state management.
- Exposed to Python via pyo3 (abi3/extension-module) and packaged with maturin (pyproject.toml).
- Key features:
  - Fast embedding / inference support (ONNX runtime and optionally libtorch via tch bindings).
  - OpenCV based image I/O & preprocessing.
  - HNSW (hnsw_rs) based ANN index for fast face matching.
  - CCTV-tracking utilities to track faces across frames and maintain per-camera state.
  - Serialization helpers for persisting indexes/metadata (bincode / serde).

---

## Files and modules

The module layout in `src/`:

- Cargo.toml
- pyproject.toml
- src/lib.rs — Python-facing API surface (pyo3 bindings).
- src/preprocess.rs — image preprocessing and tensor conversion utilities.
- src/hnsw_helper.rs — HNSW index creation, search, persistence helpers.
- src/cctv_tracker.rs — tracking primitives for CCTV streams.
- src/cctv_state.rs — per-camera / per-stream state management.
- src/utils.rs — helper utilities (conversions, JSON/time helpers).
- src/models/ — intended for model loaders and model-specific logic.
- src/scheduler/ — intended for scheduling/cleanup jobs.

(See source files for function/class level details and exact exported Python symbols.)

---

## What it does (end-to-end)

Typical flow when used from Python:
1. Import the `rust_backend` module (the compiled extension).
2. Load a face model (ONNX runtime recommended) via an exposed function.
3. Preprocess images / frames using the preprocessing helpers.
4. Run inference to obtain face embeddings.
5. Add embeddings to an HNSW index (persistable).
6. For live CCTV feeds, use tracker + state modules to:
   - Associate detections across frames using `cctv_tracker`
   - Use embeddings + ANN index to identify faces quickly
7. Persist / load HNSW indexes and metadata using serialization helpers.

---

## Building

Prerequisites:
- Rust (stable toolchain compatible with edition 2024)
- Python >= 3.10
- maturin (pip install maturin)
- Platform toolchain and libraries if building with native OpenCV/libtorch support.

Build steps:
1. cd Backend/attendance_backend/rust_extensions/rust_backend
2. maturin develop --release  # build & install into current env
3. maturin build --release   # build wheels

See `pyproject.toml` for maturin configuration (libtorch/opencv download options).

---

## Usage (high-level)

See `src/lib.rs` for the exact exported names. Example pseudocode:

```python
import rust_backend

model = rust_backend.load_model("path/to/model.onnx")
tensor = rust_backend.preprocess_image(cv_image)
embedding = model.infer(tensor)

index = rust_backend.HnswIndex(dim=embedding_dim)
index.add(id="user-123", vector=embedding)
results = index.search(embedding, k=5)

tracker = rust_backend.CctvTracker(camera_id="cam-1")
tracker.update(detections, embeddings)
state = tracker.snapshot()
```

---

## Persistence & compatibility

- HNSW indexes and metadata are serialized with serde + bincode — ensure schema compatibility when loading older indexes.
- Embedding dimensionality must match index dimensionality.

---

## Troubleshooting

- Build errors for OpenCV/libtorch: install platform dev packages or use maturin download options in pyproject.toml.
- ONNX runtime issues: check model input shapes and preprocess pipeline (see `preprocess.rs`).
- ABI / Python mismatch: rebuild with the correct Python interpreter or use abi3 builds.

---

## License

This project and the `rust_backend` crate are proprietary and protected by "All Rights Reserved" terms. See the repository LICENSE file for full text.

Summary:
- Copyright and ownership: This software (source code, binaries, documentation, and associated materials) is the exclusive property of the repository owner: Anshu-Gondi.
- Allowed use: End-users may use the distributed services or deployed software as provided by the owner (i.e., use the hosted/deployed service).
- Prohibited without explicit permission from the owner: copying, forking for redistribution, modifying, creating derivative works, reverse-engineering, publishing, sublicensing, or distributing the software or its parts.
- To request permission to use, modify, or redistribute, contact the owner via: https://github.com/Anshu-Gondi or open an issue in the repository.

This README includes a summary only — consult the LICENSE file for the full legal text.

---

## Where to look next in the code

- `src/lib.rs` — Python API and bindings
- `src/preprocess.rs` — image transform details
- `src/hnsw_helper.rs` — index and search helpers
- `src/cctv_tracker.rs` & `src/cctv_state.rs` — tracking/state logic

---

## Contact / maintainers

- Repository owner: Anshu-Gondi — https://github.com/Anshu-Gondi
