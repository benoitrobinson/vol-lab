"""Record the TUI to an asciicast, for the animation in the README.

The usual tool for this (vhs) drives a headless Chrome through the
HeadlessExperimental CDP domain, which current Chrome no longer serves: it
runs, reports success and writes nothing. So the session is driven here
instead, through a pty, with the standard library only. `agg` turns the cast
into a GIF without a browser:

    uv run python scripts/record_view.py && ./scripts/render_view_gif.sh

The keystrokes below are the ones the README documents, so the animation
cannot drift from the key table without this script failing first.
"""

import codecs
import fcntl
import json
import os
import pty
import select
import struct
import sys
import termios
import time

COLS, ROWS = 108, 34
DOWN, ESC = "\x1b[B", "\x1b"
FIRST_PAINT = "vol-lab"   # the header title: the first frame worth showing

# (seconds to wait, then the keys to send). The waits are what makes the GIF
# readable; every run finishes well inside the wait that follows it.
SCRIPT = [
    (3.5, None),      # import numpy, scipy and textual, then draw
    (2.2, DOWN),      # lessons: walk down the findings
    (2.2, DOWN * 8),  # jump to rough volatility
    (2.6, DOWN * 3),  # and on to the two market-making frictions
    (2.6, "3"),       # surface
    (0.5, "r"),       # fit SVI: the smile and the density it implies
    (5.0, "4"),       # hedge
    (0.5, "r"),       # terminal P&L and the full explain
    (5.5, "5"),       # making
    (0.5, "r"),       # three strategies on identical paths
    (6.5, "8"),       # book: lob-lab's Rust study, run on the recorded data
    (4.0, "9"),       # contracts: contract-lab's OCaml pricer on the BTC smile
    (8.0, "0"),       # exercises: predict first
    (2.5, "b"),       # commit to an answer; the real experiment grades it
    (7.0, "g"),       # the guide to the current tab
    (5.5, ESC),
    (1.0, "?"),       # the key table
    (3.5, ESC),
    (1.0, "q"),
    (1.5, None),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "vol-lab-view.cast")


def record():
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(TERM="xterm-256color", COLORTERM="truecolor",
                          LINES=str(ROWS), COLUMNS=str(COLS))
        os.execvp(sys.executable, [
            sys.executable, "-c",
            "from vollab.cli import main; raise SystemExit(main())", "view"])

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    events, start = [], time.time()
    # One decoder across reads: a chunk can end inside a multi-byte character,
    # and decoding chunks separately turned the chart blocks into U+FFFD.
    decode = codecs.getincrementaldecoder("utf-8")("replace").decode
    for wait, keys in SCRIPT:
        deadline = time.time() + wait
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], max(deadline - time.time(), 0))
            if not r:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                break
            events.append([time.time() - start, decode(chunk)])
        if keys:
            os.write(fd, keys.encode())
    os.close(fd)
    os.waitpid(pid, 0)
    return events


def main():
    events = record()
    # Drop everything before the app first paints, so the GIF does not open on
    # a second of bare cursor while numpy and textual import.
    first = next((i for i, (_, text) in enumerate(events) if FIRST_PAINT in text), 0)
    events = events[first:]
    t0 = events[0][0]

    with open(OUT, "w") as f:
        f.write(json.dumps({"version": 2, "width": COLS, "height": ROWS,
                            "timestamp": int(time.time()),
                            "env": {"TERM": "xterm-256color"}}) + "\n")
        for t, text in events:
            f.write(json.dumps([round(t - t0, 4), "o", text]) + "\n")
    print(f"{OUT}  {len(events)} events  {events[-1][0] - t0:.1f}s")


if __name__ == "__main__":
    main()
