# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Build
# Uses a Rust + OpenCV + ONNX Runtime builder image.
# Target CPU: x86-64-v2 (SSE4.2 max, no AVX) — safe on all cloud VMs.
# ══════════════════════════════════════════════════════════════════════════════
FROM rust:1.78-bookworm AS builder

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV build deps
    libopencv-dev \
    clang \
    libclang-dev \
    # LLD linker (matches .cargo/config.toml)
    lld \
    # ORT deps
    libssl-dev \
    pkg-config \
    # Python for ORT extraction
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Install ONNX Runtime at SSE4.2-safe baseline ─────────────────────────────
# ORT 1.18.0 Linux x86_64 generic build = SSE4.2 max, no AVX requirement
RUN pip3 install --break-system-packages onnxruntime==1.18.0 && \
    python3 -c " \
import onnxruntime, os, shutil, glob; \
capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'); \
libs = glob.glob(os.path.join(capi, 'libonnxruntime.so*')); \
shutil.copy(libs[0], '/usr/lib/x86_64-linux-gnu/libonnxruntime.so'); \
print('ORT installed:', libs[0])"

# ── Set ORT env so cargo build finds it ──────────────────────────────────────
ENV ORT_STRATEGY=system
ENV ORT_DYLIB_PATH=/usr/lib/x86_64-linux-gnu/libonnxruntime.so

# ── OpenCV env ────────────────────────────────────────────────────────────────
ENV OPENCV_INCLUDE_PATHS=/usr/include/opencv4
ENV OPENCV_LINK_PATHS=/usr/lib/x86_64-linux-gnu
ENV OPENCV_LINK_LIBS=opencv_core,opencv_imgcodecs,opencv_imgproc,opencv_objdetect,opencv_calib3d

# ── Copy workspace ────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY intelligence_core intelligence_core
COPY intelligence_py intelligence_py
COPY vision_engine vision_engine

# ── Build release at x86-64-v2 (SSE4.2 baseline, no AVX) ────────────────────
RUN cargo build --release \
    --package vision_engine \
    --config 'build.rustflags=["-C","target-cpu=x86-64-v2","-C","target-feature=+sse4.1,+sse4.2"]'

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime
# Minimal image — only what the binary needs to run.
# ══════════════════════════════════════════════════════════════════════════════
FROM debian:bookworm-slim AS runtime

# ── Runtime deps only ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime libs
    libopencv-core406 \
    libopencv-imgcodecs406 \
    libopencv-imgproc406 \
    libopencv-objdetect406 \
    libopencv-calib3d406 \
    # TLS
    libssl3 \
    # C++ runtime (OpenCV needs it)
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy ORT .so from builder ─────────────────────────────────────────────────
COPY --from=builder \
    /usr/lib/x86_64-linux-gnu/libonnxruntime.so \
    /usr/lib/x86_64-linux-gnu/libonnxruntime.so

# ── Copy binary from builder ──────────────────────────────────────────────────
COPY --from=builder \
    /workspace/target/release/vision_engine \
    /usr/local/bin/vision_engine

# ── Copy models ───────────────────────────────────────────────────────────────
# These must exist at these paths — mount as volume in prod
# or COPY them in if they're small enough to ship in the image.
RUN mkdir -p /app/models /app/face_db
COPY vision_engine/models /app/models

# ── Runtime environment ───────────────────────────────────────────────────────
ENV ORT_STRATEGY=system
ENV ORT_DYLIB_PATH=/usr/lib/x86_64-linux-gnu/libonnxruntime.so
ENV RUST_LOG=info
ENV FACE_DB_ROLE=writer
ENV FACE_DB_PATH=/app/face_db

# ── Health + port ─────────────────────────────────────────────────────────────
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

WORKDIR /app
CMD ["vision_engine"]