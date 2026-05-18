# =============================================================================
# Stage 1 — builder
# Installs deps (including faiss-cpu), builds the IVF+SQ8 index from
# references.json.gz. No compiler needed — faiss-cpu ships pre-built wheels.
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install ALL runtime deps here so site-packages can be copied to runtime stage
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/build_index.py .
COPY resources/ /resources/

# Build the Faiss IVF+SQ8 index (~3-5 min, outputs ~66 MB index.faiss)
# nlist=1000: 1000 clusters over 3M vectors
# nprobe=10:  search 10 clusters per query (1% of data) — fast + accurate
RUN python build_index.py \
        --references /resources/references.json.gz \
        --output-dir /data \
        --nlist 1000 \
        --nprobe 10


# =============================================================================
# Stage 2 — runtime
# Lean image: no compiler, no pip. Everything copied pre-built from builder.
# Memory budget per container:
#   Faiss IVF+SQ8 index : ~64 MB
#   labels.npy           : ~3 MB
#   Python + granian     : ~50 MB
#   Total                : ~117 MB  ✅ fits inside 160 MB limit
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy pre-compiled Python packages — no compiler needed in runtime
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/granian /usr/local/bin/granian

# Application source
COPY app/ .

# Pre-built Faiss index
COPY --from=builder /data /data

# ── Environment ───────────────────────────────────────────────────────────────
ENV INDEX_PATH=/data/index.faiss
ENV LABELS_PATH=/data/labels.npy
ENV PORT=8000
# Raise NPROBE for better recall at the cost of slightly higher latency
ENV NPROBE=10

EXPOSE 8000

# ── Granian — Rust ASGI server ────────────────────────────────────────────────
# 1 worker: index is ~64 MB; 2 workers would use ~128 MB + overhead = OOM risk
# The load balancer round-robins across api1 and api2, so we still have
# 2 parallel workers at the system level.
CMD ["sh", "-c", \
     "python -m granian \
        --interface asgi \
        --host 0.0.0.0 \
        --port $PORT \
        --workers 1 \
        --runtime-threads 1 \
        main:app"]