"""
Fraud Detection API  performance-optimized edition
"""

import os
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import faiss
import orjson
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Config
INDEX_PATH  = os.getenv("INDEX_PATH",  "/data/index.faiss")
LABELS_PATH = os.getenv("LABELS_PATH", "/data/labels.npy")
NPROBE      = int(os.getenv("NPROBE", "10"))   # clusters searched per query
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

#MCC risk table 
MCC_RISK: dict[str, float] = {
    "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
    "7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
    "5311": 0.25, "5999": 0.50,
}
MCC_DEFAULT = 0.5

#Global state 
_index:  faiss.Index | None = None
_labels: np.ndarray  | None = None

#Thread-local pre-allocated query buffer 
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

    logger.info("Loading Faiss index from %s ...", INDEX_PATH)
    idx = faiss.read_index(INDEX_PATH)
    idx.nprobe = NPROBE
    _index = idx

    logger.info("Loading labels from %s ...", LABELS_PATH)
    _labels = np.load(LABELS_PATH)

    logger.info("Ready -- %d vectors indexed  nprobe=%d", _index.ntotal, NPROBE)


async def shutdown():
    logger.info("Shutting down")


# Vectorization 
def _vectorize_into(data: dict, buf: np.ndarray) -> None:
    """Fill buf[0] in-place with the 14-dim normalised feature vector."""
    tx    = data["transaction"]
    cust  = data["customer"]
    merch = data["merchant"]
    term  = data["terminal"]
    last  = data.get("last_transaction")

    amount     = tx["amount"]
    avg_amount = cust["avg_amount"]

    requested_at = datetime.fromisoformat(tx["requested_at"].replace("Z", "+00:00"))

    b = buf[0]

    # 0 — amount
    v = amount / MAX_AMOUNT;              b[0] = v if v < 1.0 else 1.0
    # 1 — installments
    v = tx["installments"] / MAX_INSTALLMENTS; b[1] = v if v < 1.0 else 1.0
    # 2 — amount vs avg
    v = (amount / avg_amount / AMOUNT_VS_AVG_RATIO) if avg_amount > 0 else 1.0
    b[2] = v if v < 1.0 else 1.0
    # 3 — hour of day
    b[3] = requested_at.hour / 23.0
    # 4 — day of week
    b[4] = requested_at.weekday() / 6.0
    # 5 & 6 — last transaction
    if last is None:
        b[5] = -1.0; b[6] = -1.0
    else:
        last_ts = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        minutes = (requested_at - last_ts).total_seconds() / 60.0
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


#  Handlers 
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

    # 4. Score
    fraud_count = int(np.sum(_labels[ids[0]] == 1))
    score       = fraud_count / K
    approved    = score < THRESHOLD

    # 5. Respond
    return Response(
        orjson.dumps({"approved": approved, "fraud_score": score}),
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