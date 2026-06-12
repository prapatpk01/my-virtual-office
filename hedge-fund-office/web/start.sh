#!/usr/bin/env bash
# Launch Sentinel Portfolio Dashboard
PORT=${1:-8080}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Sentinel Dashboard → http://localhost:$PORT"
python3 -m http.server "$PORT" --directory "$DIR"
