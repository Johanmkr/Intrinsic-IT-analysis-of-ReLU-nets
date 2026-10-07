# Reproduction image for "Intrinsic Information Theoretic Analysis of ReLU Nets".
#
# Contains the code, the locked environment (uv.lock), MNIST and WBC, and the
# stored results, so it runs without internet. See README.md, "Docker".
#
#   docker build -t intrinsic-it-relu-nets .
#   docker run --rm intrinsic-it-relu-nets step5      # figures from the stored results
#   docker run --rm intrinsic-it-relu-nets all        # rerun every experiment
FROM ghcr.io/astral-sh/uv:0.12.21-python3.13-trixie-slim

# git: provenance.json records the commit; rsync: ./run.sh smoke.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git rsync \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

WORKDIR /app
ENV UV_FROZEN=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN ./run.sh setup

# Runnable as any user (docker run --user "$(id -u):$(id -g)"), so files
# written to mounted folders belong to the host user. Only what the pipeline
# writes is made writable (a recursive chmod of /app would copy .venv into a
# new layer); git must not count these mode changes as uncommitted edits.
# USER: PyTorch looks up the user name, and an arbitrary --user UID has no
# /etc/passwd entry (getpass reads $USER first).
ENV HOME=/tmp \
    USER=reproducer \
    MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p outputs figures latex logs .cache \
    && chmod a+rwx . configs \
    && chmod -R a+rwX outputs results figures latex logs .cache \
    && git config --system core.fileMode false

ENTRYPOINT ["./run.sh"]
CMD ["help"]
