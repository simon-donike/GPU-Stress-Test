# GPU-Stress-Test container
# Uses a CUDA runtime image so GPUs are available when launched with `--gpus`.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Default command (override in docker run)
CMD ["python", "run.py"]
