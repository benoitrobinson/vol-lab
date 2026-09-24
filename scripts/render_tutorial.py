"""Write docs/TUTORIAL.md from the guides the panel shows under g.

    uv run python scripts/render_tutorial.py

tests/test_guides.py fails when the committed file differs from this output.
"""

from pathlib import Path

from vollab.tui.exercises import EXERCISES
from vollab.tui.guides import tutorial_markdown

OUT = Path(__file__).resolve().parent.parent / "docs" / "TUTORIAL.md"

if __name__ == "__main__":
    OUT.write_text(tutorial_markdown({e.key: e.title for e in EXERCISES}))
    print(f"wrote {OUT}")
