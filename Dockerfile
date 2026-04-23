FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/{models,knowledge,cache} logs

# Expose ports
EXPOSE 8000 8001 8501

# Default command
CMD ["python", "-m", "src.interfaces.openai_api"]

# Optional: Build arguments for ARM64 build
# docker buildx build --platform linux/arm64 -t warp-agent:latest .