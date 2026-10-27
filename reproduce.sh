#!/usr/bin/env bash
# Recompute the reported numbers from the shipped outputs and verdicts, and fail if any differs from
# expected.json. No GPU, no model weights, no network. Needs python3 with numpy.
#   bash reproduce.sh              # check (exit 0 = all reproduced, 3 = a number differs, other = error)
#   bash reproduce.sh --negative   # flip one verdict per verdict file in copies; each must fail with exit 3
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS --quiet 2>/dev/null || shasum -a 256 -c SHA256SUMS --quiet
run() {  # $1 = root of a copy of this repository
  local R="$1" T; T=$(mktemp -d); trap 'rm -rf "$T"' RETURN
  (cd "$R/pilot" && python3 analyze_test.py out_test/arms_minicpm.jsonl out_test/judge/verdicts.jsonl \
      out_test/clusters.json "$T/test.json" >/dev/null &&
    python3 analyze_extra.py out_test/arms_minicpm.jsonl out_test/judge/verdicts.jsonl \
      out_test/clusters.json "$T/extra.json" >/dev/null &&
    python3 analyze_holdout.py out_holdout/arms_minicpm.jsonl out_holdout/judge/verdicts.jsonl \
      out_holdout/clusters.json "$T/holdout.json" out_holdout/gemini_arms.jsonl \
      out_holdout/judge_gemini/verdicts.jsonl >/dev/null &&
    python3 analyze_scoring.py "$T/scoring.json" >/dev/null)
  python3 - "$T" "$R/expected.json" <<'PY'
import json, os, sys
T, exp = sys.argv[1], json.load(open(sys.argv[2]))
def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d
bad = []
for name, want_all in exp.items():
    src = json.load(open(os.path.join(T, name + ".json")))
    for path, want in want_all.items():
        got = get(src, path)
        if got != want:
            bad.append(f"{name}:{path}: expected {want}, got {got}")
print("\n".join(bad) if bad else f"all {sum(map(len, exp.values()))} numbers reproduced")
sys.exit(3 if bad else 0)
PY
}
if [[ "${1:-}" == "--negative" ]]; then
  for V in out_test/judge/verdicts.jsonl out_holdout/judge/verdicts.jsonl out_holdout/judge_gemini/verdicts.jsonl; do
    C=$(mktemp -d); cp -R . "$C/"
    python3 - "$C/pilot/$V" <<'PY'
import json, sys
p = sys.argv[1]; rows = [json.loads(l) for l in open(p)]
for r in rows:                      # flip the first verdict outside the question-only arm
    if r["arm"] != "N":
        r["j1"] = not r["j1"]; break
open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
PY
    rc=0; run "$C" >/dev/null || rc=$?; rm -rf "$C"
    if [[ $rc -ne 3 ]]; then echo "NEGATIVE CONTROL FAILED for $V: exit $rc (want 3, a mismatch)"; exit 1; fi
  done
  echo "negative control OK: a flipped verdict in each verdict file changes the recomputed numbers"; exit 0
fi
run "."
