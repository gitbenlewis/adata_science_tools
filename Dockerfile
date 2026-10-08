FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    ADTL_WEB_DATA_ROOT=/data

WORKDIR /app/adata_science_tools
COPY config/requirements-web-container.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir /data \
    && chown app:app /data

COPY __init__.py ./
COPY _io/ ./_io/
COPY _preprocessing/ ./_preprocessing/
COPY _plotting/ ./_plotting/
COPY _tools/ ./_tools/
COPY web/ ./web/
COPY scripts/run_web.py ./scripts/run_web.py

USER app
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--preload", "--timeout", "120", "--access-logfile", "-", "adata_science_tools.web:create_app()"]
