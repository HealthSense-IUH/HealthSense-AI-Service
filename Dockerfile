# ========================
# Stage 1: Dependencies
# ========================
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ========================
# Stage 2: Production
# ========================
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies từ builder stage (multi-stage build giúp image nhẹ hơn)
COPY --from=builder /install /usr/local

# Copy source code
COPY ./app ./app

# Port mặc định
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Chạy server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
