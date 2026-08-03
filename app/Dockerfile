FROM python:3.12-slim

WORKDIR /app

# Install dependencies used by the service and Adaptive v12 Telegram charts.
RUN pip install --no-cache-dir websockets matplotlib

# Copy application files
COPY . .

# Create data directory
RUN mkdir -p /data

# Default environment
ENV VO_STATUS_DIR=/data \
    VO_OPENCLAW_PATH=/openclaw \
    VO_PORT=8090 \
    VO_WS_PORT=8091

EXPOSE 8090 8091

RUN chmod +x entrypoint.sh

CMD ["sh", "entrypoint.sh"]
