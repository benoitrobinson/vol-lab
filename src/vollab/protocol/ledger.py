"""Run ledger.

Every run leaves a row, including a crashed one, which is why status and
n_paths_completed exist. paths_fingerprint covers only the path-generating
inputs, so a paired comparison can refuse two runs that did not share paths.
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, ts TEXT NOT NULL,
  config_hash TEXT NOT NULL, config_toml TEXT NOT NULL, hypothesis TEXT NOT NULL,
  paths_fingerprint TEXT NOT NULL,
  git_commit TEXT NOT NULL, git_dirty INTEGER NOT NULL, git_diff_sha TEXT,
  vollab_version TEXT NOT NULL, rng_scheme_version INTEGER NOT NULL,
  engine TEXT NOT NULL, engine_build_id TEXT,
  numpy_version TEXT NOT NULL, scipy_version TEXT NOT NULL,
  python_version TEXT NOT NULL, platform TEXT NOT NULL,
  status TEXT NOT NULL, n_paths_completed INTEGER NOT NULL,
  metrics TEXT NOT NULL, artifacts TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
  runtime_s REAL NOT NULL);
CREATE INDEX IF NOT EXISTS runs_config_hash ON runs(config_hash);
CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts);
"""


class Ledger:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def insert(self, row):
        cols = ",".join(row)
        marks = ",".join(":" + c for c in row)
        self.conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", row)
        self.conn.commit()

    def all(self):
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM runs ORDER BY ts DESC")]

    def get(self, run_id):
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None
