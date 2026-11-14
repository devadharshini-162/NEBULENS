# Multi-stage Dockerfile for NEBULENS
# Stage 1: Build the React + Vite frontend
FROM oven/bun:1.1 as frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/bun.lock* ./
RUN bun install --frozen-lockfile || bun install

COPY frontend/ ./
RUN bun run build

# Stage 2: Python Backend & Static Server
FROM python:3.12-slim

WORKDIR /app

# Install minimal OS dependencies for geospatial & C++ bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, configurations, and analytical datasets
COPY backend/ ./backend/
COPY data/ ./data/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose default serving port
EXPOSE 8000

ENV ENVIRONMENT=production \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Run FastAPI backend with Uvicorn, respecting dynamically assigned PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
