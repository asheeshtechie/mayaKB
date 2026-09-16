#!/bin/sh
# End-to-end check of the Docker setup: start db + ollama, ingest the samples, search, check the citation.
set -eu
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "copy .env.example to .env first" >&2; exit 1; }

docker compose --profile ollama up -d --wait db ollama
docker compose --profile ollama run --rm ollama-pull
docker compose build app
docker compose run --rm app mayakb ingest --kb samples
docker compose run --rm app \
  mayakb search "How long do refunds take?" --kb samples --json > /tmp/mayakb-smoke.json

python3 - <<'EOF'
import json
result = json.load(open("/tmp/mayakb-smoke.json"))
top = result["chunks"][0]
assert top["citation"]["source_uri"].endswith("customer-policies.md"), top
assert "Refunds" in top["citation"]["section_path"], top
print("smoke test passed:", top["citation"]["source_uri"], top["citation"]["section_path"])
EOF
