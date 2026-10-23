#!/bin/sh
# DiabetaBot container entrypoint.
#
#   serve          start the web server and Telegram polling (default)
#   index          embed the TKPI+GI dataset into ChromaDB, then exit
#   <any command>  run it directly, so `docker run ... sh` still works
set -eu

command="${1:-serve}"

case "$command" in
    serve)
        exec uvicorn --factory app.main:create_production_app --host 0.0.0.0 --port 8000
        ;;
    index)
        shift
        exec python -m scripts.index_data "$@"
        ;;
    -*|"")
        # Flags (or nothing) belong to the server, not to a subcommand.
        exec uvicorn --factory app.main:create_production_app --host 0.0.0.0 --port 8000 "$@"
        ;;
    *)
        if command -v "$command" >/dev/null 2>&1; then
            exec "$@"
        fi
        echo "unknown command: $command" >&2
        echo "usage: docker run <image> [serve|index|<command>]" >&2
        exit 64
        ;;
esac
