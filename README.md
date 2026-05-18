  [🇺🇸 English](README.md) | [🇧🇷 Português](README.pt-BR.md) 


<div align="center">

# Rinha de Backend 2026 — Fraud Detection API - Python

<p align="center"> 
  <img src="https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" /> 
  <img src="https://img.shields.io/badge/starlette-0.41.3-black?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/granian-2.7.4-orange?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/nginx-1.27-green?style=for-the-badge&logo=nginx&logoColor=white" /> 
  <img src="https://img.shields.io/badge/faiss-ivf+sq8-red?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/orjson-rust%20powered-purple?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/docker-compose-blue?style=for-the-badge&logo=docker&logoColor=white" />
</p>

</div>

Submission for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): a fraud detection API using approximate nearest-neighbour vector search over 3 million labelled transactions.

---

## Why Python?

This project was intentionally built in Python, even knowing that the competition is heavily dominated by lower-level languages such as Rust, Go and C, which naturally have advantages in raw throughput and latency.

The goal here is not necessarily to outperform those ecosystems, but to **explore how far Python can be pushed when every millisecond matters**.

The project combines low-overhead architectural decisions, Rust-powered dependencies, memory-conscious optimizations, vector-search compression techniques and careful hot-path tuning to make Python as performant as possible within the competition constraints.

---

## The real challenge: memory, not CPU

At first, **I tried to use `hnswlib` with an HNSW index**.

The problem was discovered only after multiple local stress tests and full Docker orchestration runs: the memory footprint was fundamentally incompatible with the competition limits.

The competition rules require:

- at least 2 API instances
- 1 load balancer
- maximum total budget of:
  - 1 CPU
  - 350 MB RAM

**The original HNSW approach became impossible because HNSW stores the entire graph structure in memory for every vector.**

With:

- 3 million vectors
- 14 dimensions
- HNSW `M=8`

the index consumed approximately:

```
~721 MB RAM
```

No amount of parameter tuning could make two API containers coexist inside the allowed memory budget.

The bottleneck was not Python itself.

It was the memory model of the vector index.

---

## The solution: Faiss IVF + SQ8


**Faiss (Facebook AI Similarity Search)** is a high-performance vector similarity search library developed by Meta AI and widely used in recommendation systems, semantic search, embeddings retrieval and large-scale nearest-neighbour workloads.

For this project, the chosen index type was:

```python
faiss.IndexIVFScalarQuantizer(...)
````

which combines two complementary techniques:

* IVF (Inverted File Index)
* SQ8 (8-bit Scalar Quantization)

The reason for this choice was memory efficiency.

While HNSW provides excellent recall and latency, its graph-based structure becomes extremely memory-intensive at large scale because the full graph connectivity must remain resident in RAM.

Faiss IVF+SQ8 trades a small amount of recall for a massive reduction in memory usage while still keeping sub-millisecond query times.

This made it possible to fit 3 million vectors inside the competition's strict container memory limits.



The architecture was redesigned around two complementary Faiss techniques:

### 1. IVF — Inverted File Index

The dataset is partitioned into 1000 clusters during build time.

At query time, only 10 clusters are searched (`nprobe=10`), reducing the search space to roughly 1% of the full dataset.

Instead of:


**O(N)**


queries become approximately:


**O(N / nlist * nprobe)**


This dramatically reduces search cost while maintaining good recall.

---

### 2. SQ8 — Scalar Quantization (8-bit)

Each float32 dimension is compressed:


**float32 (4 bytes) → uint8 (1 byte)**


This provides:

* ~4× memory reduction
* minimal accuracy degradation
* measured ~97.4% recall compared to IVFFlat

Because the vectors are already normalized into `[0,1]`, scalar quantization works extremely well for this dataset.

---

## Memory impact

The migration from HNSW to Faiss IVF+SQ8 completely changed the feasibility of the architecture.

| Component           | RAM     |
| ------------------- | ------- |
| Faiss IVF+SQ8 index | ~64 MB  |
| labels.npy          | ~3 MB   |
| Python + granian    | ~50 MB  |
| Total per container | ~117 MB |

Compared to the original HNSW approach:

| Index type    | Approx RAM |
| ------------- | ---------- |
| hnswlib HNSW  | ~721 MB    |
| Faiss IVF+SQ8 | ~64 MB     |

This reduction made the competition constraints achievable.

---

## Architecture

```
Client
  │  POST /fraud-score  (port 9999)
  ▼
┌──────────────────────────────────┐
│ nginx 1.27-alpine                │
│ Round-robin load balancer        │
│ 0.20 CPU · 30 MB                 │
└────────────┬─────────────────────┘
             │ round-robin
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐     ┌─────────┐
│ api1    │     │ api2    │
│ granian │     │ granian │
│ 1 worker│     │ 1 worker│
│ 0.50CPU │     │ 0.40CPU │
│ 280 MB  │     │ 160 MB  │
└─────────┘     └─────────┘

Faiss IVF+SQ8 index pre-built during docker build
```


Total declared resources: **1.00 CPU · 350 MB RAM**


Fully compliant with the competition rules.

---

## Startup orchestration problem

After solving the memory footprint problem, another issue appeared:

`api1` started correctly, but `api2` frequently crashed while loading the same index simultaneously.

The root cause was memory pressure during concurrent index loading.

Even though the steady-state RAM usage fit comfortably inside the limit, Docker briefly experienced a memory spike while both containers loaded their own copy of the index at the same time.

The operating system still had the first container's index pages warm in memory while the second container started reading the same file.

---

## The fix: delayed startup sequencing

The solution was intentionally simple and deterministic:

`api2` waits before starting.

```yaml
command:
  [
    "sh",
    "-c",
    "sleep 15 && python -m granian ..."
  ]
```

This allows:

* api1 to fully initialize
* the OS page cache to stabilize
* I/O pressure to drop
* memory spikes to disappear

After introducing startup sequencing, both containers became stable simultaneously under the competition limits.

---

## Performance decisions

### 1. Starlette instead of FastAPI

FastAPI adds dependency injection and validation layers that cost measurable latency per request.

Using raw Starlette removes that overhead while preserving ASGI ergonomics.

---

### 2. orjson instead of stdlib json

`orjson` is Rust-based and significantly faster than Python's standard JSON implementation for both serialization and deserialization.

---

### 3. No Pydantic

The competition guarantees valid payloads.

Avoiding schema validation removes unnecessary allocations and CPU overhead.

All fields are accessed directly from raw dictionaries.

---

### 4. Thread-local preallocated NumPy buffers

Each worker allocates a single:

```python
(1, 14) float32
```

buffer and reuses it for every request.

Benefits:

* zero per-request NumPy allocations
* cache-friendly contiguous memory
* exact layout expected by Faiss

---

### 5. granian instead of uvicorn

`granian` uses a Rust runtime and consistently performs better in throughput and p99 latency than pure-Python ASGI servers.

---

### 6. Build-time index generation

The Faiss index is built during Docker build time.

Containers only perform:

```python
faiss.read_index(...)
```

during startup.

Benefits:

* no runtime training cost
* deterministic startup
* avoids massive transient RAM spikes

---

### 7. Multi-stage Docker image

The runtime image contains:

* no compiler
* no build-essential
* no pip install step

Only pre-built Python packages and the generated Faiss index are copied from the builder stage.

This reduces:

* image size
* startup complexity
* runtime dependencies

---

### 8. nginx tuned for low latency

The load balancer is configured as a pure pass-through proxy with:

* upstream keepalive
* `tcp_nodelay`
* disabled buffering
* disabled access logs

The rules explicitly forbid business logic in the load balancer.

---

## Stack

| Component        | Choice            | Why                             |
| ---------------- | ----------------- | ------------------------------- |
| ASGI server      | granian 2.7.4     | Rust runtime, lower p99 latency |
| Web framework    | Starlette 0.41.3  | Minimal overhead                |
| JSON             | orjson 3.11.9     | Rust-based JSON                 |
| Vector search    | Faiss IVF+SQ8     | Massive RAM reduction           |
| Numerics         | numpy 2.4.4       | Vectorized math                 |
| Load balancer    | nginx 1.27-alpine | Lightweight reverse proxy       |
| Containerization | Docker Compose    | Competition orchestration       |

---

## File tree

```
.
├── app/
│   ├── main.py
│   └── requirements.txt
├── scripts/
│   └── build_index.py
├── resources/
│   └── references.json.gz
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── test_vectorize.py
├── README.md
├── README.pt-BR.md
└── .gitignore
```

---

## Endpoints

| Method | Path           | Description                      |
| ------ | -------------- | -------------------------------- |
| GET    | `/ready`       | Health check                     |
| POST   | `/fraud-score` | Returns fraud decision and score |

```
```
