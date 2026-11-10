#!/usr/bin/env bash
# Nightly A/B submission: submit the arm that was not submitted last, then commit the log.
# Arms alternate between v2_skill and v1_budget (see the A/B plan in submission/LOG.md).
set -euo pipefail
cd "$(dirname "$0")/.."

last="$(grep -oE '`v[12]_(budget|skill)`' submission/LOG.md | tail -1 | tr -d '`')"
case "$last" in
  v1_budget) next=v2_skill; desc="A/B arm B: v1_budget + repo-localization skill (only change)" ;;
  *)         next=v1_budget; desc="A/B arm A: control" ;;
esac

log="results/logs/submit_$(date -u +%Y-%m-%d).log"
mkdir -p results/logs

# Kaggle allows one pending submission per team: skip the night instead of hitting a 400.
source scripts/env.sh >/dev/null
subs="$(uv run --no-sync kaggle competitions submissions -c gemma-4-developer-agent -v 2>/dev/null)"
pending="$(grep -E '^[0-9]+,' <<<"$subs" | { grep 'SubmissionStatus.PENDING' || true; } | cut -d, -f1 | paste -sd, -)"
if [[ -n "$pending" ]]; then
  echo "skipped $next: submission $pending still pending on Kaggle" | tee "$log"
  exit 0
fi
scripts/submit_bundle.sh "submission/$next" "$desc" 2>&1 | tee "$log"

git add submission/LOG.md
git commit -q -m "Log nightly A/B submission of $next

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
