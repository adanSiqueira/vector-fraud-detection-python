"""
Fraud Detection API — performance-optimized edition
"""

import os
import logging
import threading

import numpy as np
import faiss
import orjson
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

# Disable all logging — under heavy load it hurts latency
logging.disable(logging.CRITICAL)

# Config
INDEX_PATH  = os.getenv("INDEX_PATH",  "/data/index.faiss")
LABELS_PATH = os.getenv("LABELS_PATH", "/data/labels.npy")
NPROBE      = int(os.getenv("NPROBE", "5"))   # reduced from 10 — big p99 win
K           = 5
THRESHOLD   = 0.6
DIM         = 14

# Normalization constants
MAX_AMOUNT              = 10_000.0
MAX_INSTALLMENTS        = 12.0
AMOUNT_VS_AVG_RATIO     = 10.0
MAX_MINUTES             = 1_440.0
MAX_KM                  = 1_000.0
MAX_TX_COUNT_24H        = 20.0
MAX_MERCHANT_AVG_AMOUNT = 10_000.0

# MCC risk table
MCC_RISK: dict[str, float] = {
    "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
    "7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
    "5311": 0.25, "5999": 0.50,
}
MCC_DEFAULT = 0.5

# Precomputed response bodies — zero serialization per request
_RESPONSES: dict[int, bytes] = {
    score: orjson.dumps({"approved": (score / K) < THRESHOLD, "fraud_score": score / K})
    for score in range(K + 1)
}

# Global state
_index:  faiss.Index | None = None
_labels: np.ndarray  | None = None

# Thread-local pre-allocated query buffer
_tls = threading.local()

def _get_query_buf() -> np.ndarray:
    buf = getattr(_tls, "buf", None)
    if buf is None:
        buf = np.empty((1, DIM), dtype=np.float32, order="C")
        _tls.buf = buf
    return buf


# Startup / shutdown
async def startup():
    global _index, _labels

    faiss.omp_set_num_threads(1)

    idx = faiss.read_index(INDEX_PATH)
    idx.nprobe = NPROBE
    _index = idx

    _labels = np.load(LABELS_PATH)

    # Warm the index — touches pages, eliminates cold-cache P99 spikes
    dummy = np.zeros((1, DIM), dtype=np.float32)
    _index.search(dummy, 1)


async def shutdown():
    pass


# Vectorization — datetime parsed manually to avoid fromisoformat overhead
def _vectorize_into(data: dict, buf: np.ndarray) -> None:
    """Fill buf[0] in-place with the 14-dim normalised feature vector."""
    tx    = data["transaction"]
    cust  = data["customer"]
    merch = data["merchant"]
    term  = data["terminal"]
    last  = data.get("last_transaction")

    amount     = tx["amount"]
    avg_amount = cust["avg_amount"]

    ts = tx["requested_at"]
    # Manual timestamp parsing — much faster than datetime.fromisoformat
    hour    = int(ts[11:13])
    weekday = _iso_weekday(int(ts[:4]), int(ts[5:7]), int(ts[8:10]))

    b = buf[0]

    # 0 — amount
    v = amount / MAX_AMOUNT;              b[0] = v if v < 1.0 else 1.0
    # 1 — installments
    v = tx["installments"] / MAX_INSTALLMENTS; b[1] = v if v < 1.0 else 1.0
    # 2 — amount vs avg
    v = (amount / avg_amount / AMOUNT_VS_AVG_RATIO) if avg_amount > 0 else 1.0
    b[2] = v if v < 1.0 else 1.0
    # 3 — hour of day
    b[3] = hour / 23.0
    # 4 — day of week
    b[4] = weekday / 6.0
    # 5 & 6 — last transaction
    if last is None:
        b[5] = -1.0; b[6] = -1.0
    else:
        lts = last["timestamp"]
        minutes = _minutes_between(ts, lts)
        v = minutes / MAX_MINUTES;            b[5] = v if v < 1.0 else 1.0
        v = last["km_from_current"] / MAX_KM; b[6] = v if v < 1.0 else 1.0
    # 7 — km from home
    v = term["km_from_home"] / MAX_KM;    b[7] = v if v < 1.0 else 1.0
    # 8 — tx count 24h
    v = cust["tx_count_24h"] / MAX_TX_COUNT_24H; b[8] = v if v < 1.0 else 1.0
    # 9 — is_online
    b[9]  = 1.0 if term["is_online"]   else 0.0
    # 10 — card_present
    b[10] = 1.0 if term["card_present"] else 0.0
    # 11 — unknown merchant
    b[11] = 0.0 if merch["id"] in set(cust["known_merchants"]) else 1.0
    # 12 — mcc risk
    b[12] = MCC_RISK.get(merch["mcc"], MCC_DEFAULT)
    # 13 — merchant avg amount
    v = merch["avg_amount"] / MAX_MERCHANT_AVG_AMOUNT; b[13] = v if v < 1.0 else 1.0


def _iso_weekday(y: int, m: int, d: int) -> int:
    """Return weekday 0=Mon … 6=Sun using Tomohiko Sakamoto's algorithm."""
    t = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    if m < 3:
        y -= 1
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + d) % 7


def _minutes_between(ts_now: str, ts_last: str) -> float:
    """Compute elapsed minutes between two ISO-8601 strings without datetime parsing."""
    # Parse components directly from the string
    yn, mn, dn = int(ts_now[:4]),  int(ts_now[5:7]),  int(ts_now[8:10])
    hn, minn, sn = int(ts_now[11:13]), int(ts_now[14:16]), int(ts_now[17:19])

    yl, ml, dl = int(ts_last[:4]), int(ts_last[5:7]),  int(ts_last[8:10])
    hl, minl, sl = int(ts_last[11:13]), int(ts_last[14:16]), int(ts_last[17:19])

    # Convert each to seconds since a fixed epoch (ignoring timezone for delta)
    def to_seconds(y, mo, d, h, mi, s):
        # Days via a simple accumulation (good enough for deltas ≤ a few days)
        days = y * 365 + y // 4 - y // 100 + y // 400
        _mdays = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        days += _mdays[mo - 1] + d
        if mo > 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
            days += 1
        return days * 86400 + h * 3600 + mi * 60 + s

    delta = to_seconds(yn, mn, dn, hn, minn, sn) - to_seconds(yl, ml, dl, hl, minl, sl)
    return delta / 60.0


# Handlers
async def handle_ready(request: Request) -> Response:
    if _index is None or _labels is None:
        return Response(b'{"status":"loading"}', status_code=503,
                        media_type="application/json")
    return Response(
        orjson.dumps({"status": "ok", "elements": _index.ntotal}),
        media_type="application/json",
    )


async def handle_fraud_score(request: Request) -> Response:
    # 1. Parse
    data = orjson.loads(await request.body())

    # 2. Vectorize into thread-local buffer
    buf = _get_query_buf()
    _vectorize_into(data, buf)

    # 3. KNN search — Faiss releases the GIL internally
    _, ids = _index.search(buf, K)

    # 4. Score — use precomputed response bytes
    fraud_count = int(np.sum(_labels[ids[0]] == 1))

    # 5. Respond with precomputed body
    return Response(
        _RESPONSES[fraud_count],
        media_type="application/json",
    )


# App
app = Starlette(
    routes=[
        Route("/ready",       handle_ready,       methods=["GET"]),
        Route("/fraud-score", handle_fraud_score, methods=["POST"]),
    ],
    on_startup=[startup],
    on_shutdown=[shutdown],
)