#!/usr/bin/env bash
# Build, push and submit one bundle, then append a row to submission/LOG.md.
# Usage: scripts/submit_bundle.sh submission/v2_skill "short change description"
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null

bundle="${1:?bundle dir}"
desc="${2:?description}"
user=francescosimone
name="$(basename "$bundle")"
slug="g4a-${name//_/-}"

uv run --no-sync python scripts/validate_submission.py "$bundle" >/dev/null
build_out="$(uv run --no-sync python scripts/build_submission_notebook.py "$bundle" --kaggle-user "$user")"
sha="$(sed -n 's/.*sha256 \([0-9a-f]*\)).*/\1/p' <<<"$build_out")"

push_out="$(uv run --no-sync kaggle kernels push -p "submission/build/kernels/$slug" 2>&1)"
version="$(sed -n 's/.*Kernel version \([0-9]*\) successfully pushed.*/\1/p' <<<"$push_out")"
[[ -n "$version" ]] || { echo "$push_out"; exit 1; }

for _ in $(seq 1 60); do
  status="$(uv run --no-sync kaggle kernels status "$user/$slug" 2>&1 | tail -1)"
  case "$status" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "$status"; exit 1 ;;
  esac
  sleep 20
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "dry run: kernel $user/$slug version $version complete, bundle $name sha256 $sha; not submitted"
  exit 0
fi

# CSV output, keeping only data rows: the CLI prints an "outdated version" warning on stdout.
top_ref() { uv run --no-sync kaggle competitions submissions -c gemma-4-developer-agent -v 2>/dev/null | grep -E '^[0-9]+,' | head -1 | cut -d, -f1; }
before="$(top_ref)"
uv run --no-sync kaggle competitions submit gemma-4-developer-agent -k "$user/$slug" -f submission.zip \
  -v "$version" -m "$name sha256 $sha: $desc" 2>&1 | grep -v "outdated" || true
# The CLI exits 0 even when the daily quota is used up: accept only if a new submission appears.
ref="$(top_ref)"
[[ -n "$ref" && "$ref" != "$before" ]] || { echo "submission not accepted (top ref still $before)"; exit 1; }
printf '| %s | %s | `%s` | %s | %s | in attesa | |\n' "$(date -u '+%Y-%m-%d %H:%M')" "$ref" "$name" "$sha" "$desc" >> submission/LOG.md
echo "submitted $name ($sha) as $ref"
