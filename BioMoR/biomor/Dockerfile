FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 1. Environment Setting
# Prevent apt-get from prompting for user input
ENV DEBIAN_FRONTEND=noninteractive
# Ensure Python print statements are instantly piped out to docker logs
ENV PYTHONUNBUFFERED=1
# Suppress the PyTorch OpenMP conflict on some Windows hosts
ENV KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /workspace

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Python Environment
# Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Codebase
# When mapped with docker-compose, this allows local Windows edits to reflect immediately
COPY . /workspace/

# Default executable command
CMD ["python", "test_run.py"]
