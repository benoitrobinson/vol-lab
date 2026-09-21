# Reproduce every number in REPORT.md from a pinned operating system.
#
# uv.lock pins Python packages, but not the compiler, libm or CPU features, and
# those matter here: the C++ engine and the NumPy reference agree only to a
# tolerance because inverse-CDF normals are not bit-portable across libms, and
# the golden test had to be loosened for exactly that reason. Pinning the OS is
# the next layer up.
#
#   docker build -t vol-lab .
#   docker run --rm vol-lab                        # correctness suite
#   docker run --rm vol-lab pytest -m slow         # reproduce the findings
#   docker run --rm -v "$PWD/artifacts:/lab/artifacts" vol-lab \
#       python scripts/report.py                   # regenerate, writing to the host
#
# Context: running the original authors' own code on the original data
# reproduces a published finance result exactly only 52% of the time
# (Perignon et al., Review of Financial Studies 37(11), 2024). Everything in
# this image exists to make that number 100% for this repository.

FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /lab

# Dependency layer first, so source edits do not trigger a full rebuild.
COPY pyproject.toml uv.lock CMakeLists.txt ./
COPY cpp ./cpp
COPY src ./src
RUN uv sync --frozen

COPY . .
RUN uv sync --frozen --reinstall-package vollab \
    && uv run python -c "from vollab import _core; print('C++ engine:', _core.uniforms(7, 0, 2))"

ENV VOLLAB_HOME=/lab/.vollab
ENTRYPOINT ["uv", "run"]
CMD ["pytest", "-q"]
