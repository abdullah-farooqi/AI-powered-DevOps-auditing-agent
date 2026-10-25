# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install system dependencies (git is required for shallow repository cloning)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set up non-root user and home directory
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY agent.py agent_loop.py llm.py prompts.py report.py tools.py main.py ./
COPY schemas/ ./schemas/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "agent.py"]
CMD ["--help"]
