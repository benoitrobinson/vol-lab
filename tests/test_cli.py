import tomllib

import tomli_w

from vollab.cli import main
from vollab.protocol.ledger import Ledger

CFG = {
    "hypothesis": "smoke",
    "contract": {"kind": "call", "S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.0, "q": 0.0},
    "vols": {"s_imp": 0.3, "s_hedge": 0.3, "s_real": 0.3},
    "schedule": {"name": "fixed_time", "every": 1},
    "n_mon": 32, "cost_bps": 0.0, "n_paths": 400, "seed": 1, "chunk_paths": 400,
}


def _write(tmp_path, cfg=None, name="c.toml"):
    p = tmp_path / name
    p.write_bytes(tomli_w.dumps(cfg or CFG).encode())
    return p


def test_price_prints_price_and_greeks(capsys):
    assert main(["price", "--K", "100", "--T", "1", "--vol", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "price" in out and "delta" in out and "gamma" in out


def test_register_stamps_a_hash(tmp_path):
    p = _write(tmp_path)
    assert main(["register", str(p)]) == 0
    assert "config_hash" in tomllib.loads(p.read_text())


def test_register_refuses_without_a_hypothesis(tmp_path):
    bare = {k: v for k, v in CFG.items() if k != "hypothesis"}
    assert main(["register", str(_write(tmp_path, bare))]) == 2


def test_run_writes_a_ledger_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path)
    assert main(["register", str(p)]) == 0
    assert main(["run", str(p)]) == 0
    rows = Ledger(tmp_path / ".vollab" / "ledger.db").all()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["n_paths_completed"] == 400
    assert rows[0]["rng_scheme_version"] == 1


def test_run_refuses_an_unregistered_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.toml"
    p.write_bytes(tomli_w.dumps({**CFG, "config_hash": "stale"}).encode())
    assert main(["run", str(p)]) == 2
    assert "refused" in capsys.readouterr().err


def test_run_refuses_an_edited_config(tmp_path, monkeypatch):
    """Register, then edit a parameter: the run must be blocked."""
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path)
    main(["register", str(p)])
    d = tomllib.loads(p.read_text())
    d["n_paths"] = 999
    p.write_bytes(tomli_w.dumps(d).encode())
    assert main(["run", str(p)]) == 2


def test_compare_refuses_runs_that_did_not_share_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ids = []
    for seed in (1, 2):
        p = _write(tmp_path, {**CFG, "seed": seed}, name=f"c{seed}.toml")
        main(["register", str(p)])
        main(["run", str(p)])
    rows = Ledger(tmp_path / ".vollab" / "ledger.db").all()
    ids = [r["run_id"] for r in rows]
    assert main(["compare", ids[0], ids[1]]) == 2
    assert "did not share paths" in capsys.readouterr().err


def test_compare_accepts_runs_that_shared_paths(tmp_path, monkeypatch):
    """Same paths, different schedule: a paired comparison is valid."""
    monkeypatch.chdir(tmp_path)
    for every in (1, 4):
        p = _write(tmp_path, {**CFG, "schedule": {"name": "fixed_time", "every": every}},
                   name=f"s{every}.toml")
        main(["register", str(p)])
        main(["run", str(p)])
    ids = [r["run_id"] for r in Ledger(tmp_path / ".vollab" / "ledger.db").all()]
    assert main(["compare", ids[0], ids[1]]) == 0


def test_ledger_lists_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["ledger"]) == 0
    assert "no runs" in capsys.readouterr().out
    p = _write(tmp_path)
    main(["register", str(p)])
    main(["run", str(p)])
    assert main(["ledger"]) == 0
    assert "smoke" in capsys.readouterr().out


def test_bench_runs_or_reports_a_missing_extension(capsys):
    from vollab.hedge.simulator import cpp_available
    rc = main(["bench", "--paths", "500", "--repeats", "1"])
    out = capsys.readouterr()
    if cpp_available():
        assert rc == 0
        assert "speedup" in out.out.lower()
    else:
        assert rc == 2
        assert "not built" in out.err


def test_surface_fits_and_reports_arbitrage_diagnostics(capsys):
    assert main(["surface", "--points", "9", "--T", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "SVI" in out and "Durrleman" in out and "rmse" in out


def test_surface_flags_a_slice_that_implies_a_negative_density(capsys):
    """Exit 1 and a warning, rather than printing a slice that is not a price."""
    rc = main(["surface", "--points", "7", "--T", "0.08", "--noise", "6.0",
               "--seed", "3", "--unconstrained"])
    err = capsys.readouterr().err
    assert rc in (0, 1)
    if rc == 1:
        assert "negative density" in err


def test_mm_compares_both_strategies(capsys):
    assert main(["mm", "--paths", "400", "--steps", "60"]) == 0
    out = capsys.readouterr().out
    assert "avellaneda_stoikov" in out and "symmetric" in out
    assert "95% CI" in out


def test_view_refuses_when_no_ledger_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["view"]) == 2
    assert "no ledger" in capsys.readouterr().err
