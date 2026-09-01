"""Strict data contracts used by the evaluation controller."""

from .benchmark import (
    BenchmarkItem,
    CANONICAL_BENCHMARK_SIZE,
    SMOKE_SIZES,
    load_benchmark,
    smoke_items,
)
from .jsonio import StrictJsonError, load_json_strict, load_jsonl_strict
from .receipts import (
    BATCH_RECEIPT_SCHEMA_VERSION,
    BatchReceiptError,
    LEAKAGE_ALGORITHM_SHA256,
    LEAKAGE_ALGORITHM_VERSION,
    LEAKAGE_AUDIT_SCHEMA_VERSION,
    REQUIRED_INPUT_ROLES,
    create_batch_receipt,
    load_and_validate_batch_receipt,
    validate_batch_id,
)

__all__ = [
    "BATCH_RECEIPT_SCHEMA_VERSION",
    "BatchReceiptError",
    "BenchmarkItem",
    "CANONICAL_BENCHMARK_SIZE",
    "LEAKAGE_ALGORITHM_SHA256",
    "LEAKAGE_ALGORITHM_VERSION",
    "LEAKAGE_AUDIT_SCHEMA_VERSION",
    "REQUIRED_INPUT_ROLES",
    "SMOKE_SIZES",
    "StrictJsonError",
    "create_batch_receipt",
    "load_and_validate_batch_receipt",
    "load_benchmark",
    "load_json_strict",
    "load_jsonl_strict",
    "smoke_items",
    "validate_batch_id",
]
