# Keep Python and uv aligned with .python-version and pyproject.toml.
FROM ghcr.io/astral-sh/uv:0.11.24 AS uv
FROM python:3.11.11-slim-bookworm
COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# The lockfile supplies CUDA-enabled torch and its CUDA runtime libraries.
COPY pyproject.toml uv.lock .python-version README.md /app/
RUN uv sync --locked --no-dev --no-cache

COPY run.py /app/run.py
COPY utils /app/utils
COPY docs /app/docs
COPY LICENSE /app/LICENSE

# Starting a container must not start a stress workload.
CMD ["python", "run.py", "--help"]
