"""Read-only viewer over recorded runs.

It never runs a simulation. Keeping the viewer read-only is what allows the
engine to have no interactive code paths and stay fully testable headless.
"""

import json

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from vollab.protocol.ledger import Ledger


def format_run(row):
    """Plain text for one run. Separated from the widget so it is testable."""
    m = json.loads(row["metrics"])
    lines = [
        f"run        {row['run_id'][:12]}",
        f"when       {row['ts']}",
        f"config     {row['config_hash'][:12]}",
        f"paths fp   {row['paths_fingerprint'][:12]}",
        f"engine     {row['engine']}  rng scheme v{row['rng_scheme_version']}",
        f"status     {row['status']}  ({row['n_paths_completed']} paths)",
        f"git        {row['git_commit'][:8] or 'none'}"
        f"{'  DIRTY TREE' if row['git_dirty'] else ''}",
        f"versions   numpy {row['numpy_version']}  scipy {row['scipy_version']}"
        f"  python {row['python_version']}",
        f"runtime    {row['runtime_s']:.2f}s",
        "",
        "hypothesis",
        f"  {row['hypothesis']}",
        "",
        "metrics",
    ]
    for k in sorted(m):
        lines.append(f"  {k:20s} {m[k]:.6f}" if isinstance(m[k], float)
                     else f"  {k:20s} {m[k]}")
    return "\n".join(lines)


class ViewerApp(App):
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    DataTable { width: 45%; border: solid $accent; }
    #detail { width: 1fr; border: solid $accent; padding: 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("j,down", "next", "next run"),
        Binding("k,up", "prev", "previous run"),
    ]

    def __init__(self, db_path):
        super().__init__()
        self.rows = Ledger(db_path).all()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield DataTable(cursor_type="row")
            with VerticalScroll(id="detail"):
                yield Static(id="detail-text")
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("when", "run", "config", "status")
        for r in self.rows:
            table.add_row(r["ts"], r["run_id"][:8], r["config_hash"][:8], r["status"])
        self.title = "vol-lab"
        self.sub_title = f"{len(self.rows)} runs"
        if self.rows:
            self._show(0)

    def _show(self, index):
        if 0 <= index < len(self.rows):
            self.query_one("#detail-text", Static).update(format_run(self.rows[index]))

    def on_data_table_row_highlighted(self, event):
        self._show(event.cursor_row)

    def action_next(self):
        self.query_one(DataTable).action_cursor_down()

    def action_prev(self):
        self.query_one(DataTable).action_cursor_up()
