"""Canonical hashing of a configuration.

Canonicalisation is pinned here (sorted keys, compact separators, ASCII, no
NaN) so the hash is stable across TOML writers. Without that the stored hash
would drift on cosmetic edits.
"""

import hashlib
import json


def canonical_bytes(d):
    """The stored hash field never participates, or verifying it is circular."""
    clean = {k: v for k, v in d.items() if k != "config_hash"}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def config_hash(d):
    return hashlib.sha256(canonical_bytes(d)).hexdigest()
