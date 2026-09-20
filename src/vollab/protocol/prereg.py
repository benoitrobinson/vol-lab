"""Pre-registration.

The config file carries its own hash. Running recomputes the hash over the
canonical form and refuses when the two differ, so editing a parameter without
re-registering blocks the run. Hashing the file and comparing it against the
file it was hashed from would check nothing.
"""

import tomllib
from pathlib import Path

from vollab.protocol.hashing import config_hash


class HashMismatch(Exception):
    pass


def load_registered(path):
    d = tomllib.loads(Path(path).read_text())
    stored = d.get("config_hash")
    if stored is None:
        raise HashMismatch(f"{path} has no config_hash; register it first")
    actual = config_hash(d)
    if stored != actual:
        raise HashMismatch(
            f"{path} changed since registration: stored {stored[:12]}, "
            f"actual {actual[:12]}"
        )
    if not d.get("hypothesis"):
        raise ValueError("config must declare a prose hypothesis before it can run")
    return d
