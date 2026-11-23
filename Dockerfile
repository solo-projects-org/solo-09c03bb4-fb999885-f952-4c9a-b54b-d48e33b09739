FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DIV_DB=/data/split.sqlite3

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py stats.py ./
COPY templates ./templates
COPY static ./static

RUN mkdir -p /data && chown -R 1000:1000 /data

USER 1000:1000

EXPOSE 8000

CMD ["sh", "-c", "python -c 'import app; app.bootstrap()' && exec gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 --timeout 60 --access-logfile - --no-control-socket app:app"]
