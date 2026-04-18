# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Build
# tract is pure Rust — no ORT install step needed at all.
# ══════════════════════════════════════════════════════════════════════════════
FROM rust:1.78-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    clang \
    libclang-dev \
    lld \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# OpenCV env
ENV OPENCV_INCLUDE_PATHS=/usr/include/opencv4
ENV OPENCV_LINK_PATHS=/usr/lib/x86_64-linux-gnu
ENV OPENCV_LINK_LIBS=opencv_core,opencv_imgcodecs,opencv_imgproc,opencv_objdetect,opencv_calib3d

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY intelligence_core intelligence_core
COPY intelligence_py intelligence_py
COPY vision_engine vision_engine

# Build at SSE4.2 baseline — tract respects this automatically
RUN cargo build --release \
    --package vision_engine \
    --config 'build.rustflags=["-C","target-cpu=x86-64-v2","-C","target-feature=+sse4.1,+sse4.2"]'

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime — much smaller now, no ORT .so to carry
# ══════════════════════════════════════════════════════════════════════════════
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 \
    libopencv-imgcodecs406 \
    libopencv-imgproc406 \
    libopencv-objdetect406 \
    libopencv-calib3d406 \
    libssl3 \
    libstdc++6 \
    libgomp1 \
    # curl for HEALTHCHECK
    curl \
    && rm -rf /var/lib/apt/lists/*

# No ORT .so copy step — tract is statically linked into the binary
COPY --from=builder \
    /workspace/target/release/vision_engine \
    /usr/local/bin/vision_engine

RUN mkdir -p /app/models /app/face_db
COPY vision_engine/models /app/models

ENV RUST_LOG=info
ENV FACE_DB_ROLE=writer
ENV FACE_DB_PATH=/app/face_db

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

WORKDIR /app
CMD ["vision_engine"]