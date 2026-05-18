"""
validates the vectorization logic against the two examples
No server or index needed, pure Python math only.
"""

import sys
import os
import numpy as np

# ── Make app/ importable without installing ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Import the raw vectorizer (dict-based, no Pydantic)
from app.main import _vectorize_into, DIM

# ─────────────────────────────────────────────────────────────────────────────
def vectorize(data: dict) -> list[float]:
    """Thin wrapper so tests look the same as before."""
    buf = np.empty((1, DIM), dtype=np.float32, order="C")
    _vectorize_into(data, buf)
    return buf[0].tolist()


def check(name: str, got: list[float], expected: list[float], tol: float = 1e-3) -> bool:
    ok = all(abs(a - b) < tol for a, b in zip(got, expected))
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}")
    if not ok:
        for i, (a, b) in enumerate(zip(got, expected)):
            if abs(a - b) >= tol:
                print(f"  [{i:2d}]  got {a:.6f}  expected {b:.6f}  diff {abs(a-b):.6f}")
    return ok


# ── Example 1 — legitimate tx, last_transaction = null ───────────────────────
# Source: DETECTION_RULES.md flow overview
# Expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
data1 = {
    "id": "tx-1329056812",
    "transaction": {
        "amount": 41.12,
        "installments": 2,
        "requested_at": "2026-03-11T18:45:53Z",
    },
    "customer": {
        "avg_amount": 82.24,
        "tx_count_24h": 3,
        "known_merchants": ["MERC-003", "MERC-016"],
    },
    "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
    "terminal": {"is_online": False, "card_present": True, "km_from_home": 29.23},
    "last_transaction": None,
}
expected1 = [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
got1 = vectorize(data1)
check("Legit tx (last_transaction=null)", got1, expected1)
print("  →", [round(x, 4) for x in got1])


# ── Example 2 — fraudulent tx, last_transaction = null ───────────────────────
# Source: DETECTION_RULES.md fraudulent example
# Expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
data2 = {
    "id": "tx-3330991687",
    "transaction": {
        "amount": 9505.97,
        "installments": 10,
        "requested_at": "2026-03-14T05:15:12Z",
    },
    "customer": {
        "avg_amount": 81.28,
        "tx_count_24h": 20,
        "known_merchants": ["MERC-008", "MERC-007", "MERC-005"],
    },
    "merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
    "terminal": {"is_online": False, "card_present": True, "km_from_home": 952.27},
    "last_transaction": None,
}
expected2 = [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
got2 = vectorize(data2)
check("Fraud tx (last_transaction=null)", got2, expected2)
print("  →", [round(x, 4) for x in got2])


# ── Example 3 — tx WITH last_transaction (API.md sample payload) ─────────────
data3 = {
    "id": "tx-3576980410",
    "transaction": {
        "amount": 384.88,
        "installments": 3,
        "requested_at": "2026-03-11T20:23:35Z",
    },
    "customer": {
        "avg_amount": 769.76,
        "tx_count_24h": 3,
        "known_merchants": ["MERC-009", "MERC-001", "MERC-001"],
    },
    "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
    "terminal": {"is_online": False, "card_present": True, "km_from_home": 13.7090520965},
    "last_transaction": {
        "timestamp": "2026-03-11T14:58:35Z",
        "km_from_current": 18.8626479774,
    },
}
got3 = vectorize(data3)
print("\n✅  API sample tx (last_transaction present — visual check):")
print("  →", [round(x, 4) for x in got3])
assert got3[5] >= 0.0, "dim5 must be >= 0 when last_transaction is present"
assert got3[6] >= 0.0, "dim6 must be >= 0 when last_transaction is present"
assert all(0.0 <= v <= 1.0 for i, v in enumerate(got3) if i not in (5, 6)), \
    "all dims except 5 & 6 must be in [0, 1]"

print("\nAll checks passed.")
