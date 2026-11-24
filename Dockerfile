# Pinned CUDA runtime and PyTorch base; launch with NVIDIA Container Toolkit.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-docker.txt

# Copy only application sources and documentation needed in the image.
COPY run.py /app/run.py
COPY utils /app/utils
COPY docs /app/docs
COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

# Starting a container must not start a stress workload.
CMD ["python", "run.py", "--help"]
