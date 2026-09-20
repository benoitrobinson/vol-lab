"""Test isolation.

VOLLAB_HOME redirects the run ledger, and a developer who exports it in their
shell would otherwise have every ledger test write to their real lab instead of
to tmp_path. Those tests then fail for a reason that has nothing to do with the
code. Clearing it by default makes the suite independent of the environment it
runs in; the one test that exercises the variable sets it explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_vollab_home(monkeypatch):
    monkeypatch.delenv("VOLLAB_HOME", raising=False)
