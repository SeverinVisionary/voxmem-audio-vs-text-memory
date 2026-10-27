"""Blind judging of dev-pilot answers (PREREGISTRATION §6).

    python judge.py ARMS_JSONL OUT_DIR

Each distinct (question, response) pair is judged once, without arm labels, in a seeded shuffled order, by
`gemini-3.8-flash-high` via `agy` with the VoxMem answerable-judge prompt. A deterministic matcher runs in
parallel. Every judge/deterministic disagreement plus a seeded 10% sample per arm goes to a second judge
(`claude-opus-5-5`, low effort, via `claude -p`), also blind. Prompts and raw outputs are kept in OUT_DIR (resumable).
Writes OUT_DIR/verdicts.jsonl: one line per arms row with judge1, det, judge2 (if sampled).
"""
import collections
import csv
import json
import os
import re
import subprocess
import sys

from common import Q2, hkey

RULES = (Q2 / "phase0/out/vox_evaluation__answerable_judge_prompt.txt").read_text()
J1, J2 = os.environ.get("Q2_J1", "gemini-3.8-flash-high"), os.environ.get("Q2_J2", "claude-opus-5-5")          # J2 via `claude -p` (low effort by default)
BATCH = int(os.environ.get("Q2_JUDGE_BATCH", 40))   # cached batch files are only valid for the size that wrote them


def latest_rows(path):
    """Last row per (qid, arm): a retried row supersedes the failed one."""
    rows = {}
    for l in open(path):
        r = json.loads(l)
        rows[(r["qid"], r["arm"])] = r
    return [r for r in rows.values() if "error" not in r]


def norm(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def det(answer, gold):
    """Deterministic check: normalised gold equals, or (for short answers) is contained in, the response."""
    a, g = norm(answer), norm(gold)
    return a == g or (len(a.split()) <= len(g.split()) + 6 and f" {g} " in f" {a} ")


def call(model, items, path):
    if os.path.exists(path):
        v = json.load(open(path))["verdicts"]
        assert set(v) == {it["id"] for it in items}, f"{path} was written for other items (batch size or row set changed)"
        return v
    body = "\n\n".join(f"### Item {it['id']}\nFINAL QUESTION: {it['question']}\nREFERENCE ANSWER: {it['gold']}\n"
                       f"EXPECTED ANSWER TYPE: {it['answer_type']}\nMODEL RESPONSE: {it['response']}" for it in items)
    prompt = (RULES + "\n\nYou are given several independent items below. Judge each one separately by the rules "
              "above. Return a JSON array with one object per item: {\"id\": <item id>, \"verdict\": \"correct\" "
              "or \"incorrect\"}. Return only the JSON array.\n\n" + body)
    for attempt in range(3):
        cmd = (["claude", "-p", "--model", model, prompt] if model.startswith("claude") else
               ["agy", "--model", model, "--output-format", "text", "--disable-slash-commands", "-p", prompt])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:                # agy occasionally hangs; ask again
            print(f"{model} batch {path} attempt {attempt + 1} timed out", flush=True)
            continue
        m = re.search(r"\[.*\]", r.stdout, re.S)
        try:
            # judges sometimes return the numeric ids as JSON numbers, which drops leading zeros
            want = {it["id"].lstrip("0"): it["id"] for it in items}
            txt = re.sub(r'("id"\s*:\s*)(\d+)', r'\1"\2"', m.group(0))   # 0-led bare ids are invalid JSON
            got = {str(d["id"]).lstrip("0"): d["verdict"] for d in json.loads(txt)}
            v = {want[k]: got[k] for k in got if k in want}
            if set(got) == set(want):
                json.dump(dict(model=model, prompt=prompt, raw=r.stdout, verdicts=v), open(path, "w"), indent=1)
                return v
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
            pass
        print(f"{model} batch {path} attempt {attempt + 1} unusable: {r.stdout[-300:]} {r.stderr[-300:]}", flush=True)
    raise RuntimeError(f"judge {model} failed on {path}")


def run(model, items, out, tag):
    os.makedirs(out, exist_ok=True)
    items = sorted(items, key=lambda it: hkey("judge", tag, it["id"]))
    v = {}
    for i in range(0, len(items), BATCH):
        v.update(call(model, items[i:i + BATCH], os.path.join(out, f"{tag}_{i // BATCH:03d}.json")))
    return v


def main():
    arms, out = sys.argv[1], sys.argv[2]
    meta = {r["question_id"]: r for r in csv.DictReader(open(Q2 / "phase0/out/vox_items_8k.csv"))}
    rows = latest_rows(arms)
    uniq = {}
    for r in rows:
        k = f"{r['qid']}|{r['answer'].strip()}"
        if k not in uniq:
            uniq[k] = dict(id=f"{int(hkey('id', k) * 1e12):012d}", question=meta[r["qid"]]["question_text"], gold=r["gold"],
                           answer_type=r["answer_type"], response=r["answer"].strip())
    v1 = run(J1, list(uniq.values()), out, "j1")
    second = set()
    by_arm = collections.defaultdict(list)
    for r in rows:
        k = f"{r['qid']}|{r['answer'].strip()}"
        by_arm[r["arm"]].append(k)
        if (v1[uniq[k]["id"]] == "correct") != det(r["answer"], r["gold"]):
            second.add(k)
    for arm, ks in by_arm.items():                       # seeded 10% per arm
        second |= set(sorted(ks, key=lambda k: hkey("sample", arm, k))[:max(1, round(0.1 * len(ks)))])
    v2 = run(J2, [uniq[k] for k in sorted(second)], out, "j2")
    with open(os.path.join(out, "verdicts.jsonl"), "w") as f:
        for r in rows:
            k = f"{r['qid']}|{r['answer'].strip()}"
            iid = uniq[k]["id"]
            f.write(json.dumps(dict(qid=r["qid"], arm=r["arm"], j1=v1[iid] == "correct", det=det(r["answer"], r["gold"]),
                                    j2=(v2[iid] == "correct") if iid in v2 else None)) + "\n")
    print(f"{len(rows)} rows, {len(uniq)} distinct responses, {len(second)} to second judge")


if __name__ == "__main__":
    main()
