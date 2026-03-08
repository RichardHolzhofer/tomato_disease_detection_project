# Use Python 3.12 as the base image (matches pyproject.toml)
FROM python:3.12-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies (often required for ML/image processing libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Remove the CUDA specific configurations from pyproject.toml
# to force installation of PyTorch CPU wheels instead
RUN sed -i '/\[\[tool.uv.index\]\]/,$d' pyproject.toml

# Install dependencies using uv into a virtual environment,
# explicitly querying the PyTorch CPU index first.
RUN uv venv && \
    uv pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    .

# Copy the rest of the application code
# (Relies on .dockerignore to exclude .venv, notebooks, etc.)
COPY . /app/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Expose the default Streamlit port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
