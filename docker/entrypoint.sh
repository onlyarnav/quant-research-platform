#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-api}"

case "$CMD" in
  api)
    shift || true
    exec uvicorn src.dashboard.api.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers "${API_WORKERS:-1}" \
      --proxy-headers \
      --forwarded-allow-ips='*' \
      "$@"
    ;;
  dashboard)
    shift || true
    exec streamlit run src/dashboard/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      "$@"
    ;;
  scheduler)
    shift || true
    exec python scripts/run_scheduler.py "$@"
    ;;
  migrate)
    shift || true
    exec alembic upgrade head "$@"
    ;;
  shell)
    shift || true
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
