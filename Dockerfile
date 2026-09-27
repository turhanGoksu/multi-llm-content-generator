# syntax=docker/dockerfile:1

# ---- builder: install dependencies into an isolated virtualenv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements first: this layer is rebuilt only when dependencies change,
# not on every code edit.
COPY requirements.txt .
RUN pip install -r requirements.txt


# ---- runtime: only the virtualenv and the application code ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Explicit paths instead of `COPY . .`, so nothing else (e.g. .env) can slip in.
COPY config/ config/
COPY app/ app/

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
