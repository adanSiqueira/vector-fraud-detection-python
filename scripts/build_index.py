"""
Gotta be run ONCE at Docker build time.

Reads resources/references.json.gz and builds a Faiss IVF+SQ8 index:
  /data/index.faiss  — Faiss binary index (~66 MB on disk, ~64 MB in RAM)
  /data/labels.npy   — int8 numpy array (1=fraud, 0=legit, ~3 MB)
"""

import argparse
import gzip
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DIM = 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Faiss IVF+SQ8 index from references.json.gz")
    p.add_argument("--references",  default="resources/references.json.gz")
    p.add_argument("--output-dir",  default="/data")
    p.add_argument("--nlist",       type=int, default=1000,
                   help="Number of IVF clusters. More = faster queries, less recall.")
    p.add_argument("--nprobe",      type=int, default=10,
                   help="Clusters to search per query.")
    return p.parse_args()


def load_references(path: str) -> list:
    logger.info("Reading %s ...", path)
    t0 = time.time()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d records in %.1f s", len(data), time.time() - t0)
    return data


def build(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_references(args.references)
    n = len(data)

    logger.info("Converting %d records to numpy arrays ...", n)
    vectors   = np.empty((n, DIM), dtype=np.float32)
    label_arr = np.empty(n,        dtype=np.int8)

    for i, rec in enumerate(data):
        vectors[i]   = rec["vector"]
        label_arr[i] = 1 if rec["label"] == "fraud" else 0
        if (i + 1) % 500_000 == 0:
            logger.info("  converted %d / %d ...", i + 1, n)

    del data
    gc.collect()

    logger.info(
        "Building Faiss IVF%d+SQ8 index (dim=%d, n=%d) ...",
        args.nlist, DIM, n,
    )
    t0 = time.time()

    faiss.omp_set_num_threads(os.cpu_count() or 4)

    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        DIM,
        args.nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_L2,
    )

    logger.info("  training on %d vectors ...", n)
    index.train(vectors)

    logger.info("  adding %d vectors ...", n)
    index.add(vectors)

    logger.info("Index built in %.1f s  (%d vectors)", time.time() - t0, index.ntotal)

    index_path  = os.path.join(args.output_dir, "index.faiss")
    labels_path = os.path.join(args.output_dir, "labels.npy")

    faiss.write_index(index, index_path)
    np.save(labels_path, label_arr)

    logger.info("index.faiss -> %.1f MB", os.path.getsize(index_path)  / 1e6)
    logger.info("labels.npy  -> %.1f MB", os.path.getsize(labels_path) / 1e6)
    logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.references):
        logger.error("References file not found: %s", args.references)
        sys.exit(1)
    build(args)