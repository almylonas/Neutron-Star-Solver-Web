# Use Python 3.11 for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that numba might need
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first
COPY requirements.txt .

# Install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Compile the Numba kernels once at build time (cached in __pycache__),
# so the container starts serving immediately
RUN python -c "import main"

# Production server. One worker keeps a single shared result cache;
# threads handle concurrent requests (the solver itself is serialised).
ENV PORT=5000
CMD gunicorn -w 1 --threads 4 -b 0.0.0.0:${PORT} main:app
