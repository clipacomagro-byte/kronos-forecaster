FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Pull Kronos model code from repo
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /tmp/kronos \
    && cp -r /tmp/kronos/model /app/model \
    && rm -rf /tmp/kronos

# Python deps — CPU-only torch for smaller image; swap for gpu variant if needed
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace model cache lives in /app/.cache so Railway volume can persist it
ENV HF_HOME=/app/.cache
ENV PORT=8080

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300 --log-level info
