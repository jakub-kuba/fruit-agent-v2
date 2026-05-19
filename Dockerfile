# Dockerfile
FROM python:3.12-slim

# Install curl for healthcheck and clean apt cache to keep image small
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --no-dev

# Copy application code
COPY agent/ agent/
COPY db/ db/
COPY data/ data/
COPY main.py .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]