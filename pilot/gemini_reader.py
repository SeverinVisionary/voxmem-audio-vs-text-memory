"""Strong-reader ceiling (PREREG_HOLDOUT §4): Gemini through the `agy` CLI on exported plans. Resumable.

    python gemini_reader.py PLANS_DIR OUT_JSONL [--model gemini-3.8-flash-high]
    python gemini_reader.py PLANS_DIR OUT_JSONL --gold GOLD_JSONL --deny DIR [--deny DIR ...]   # sealed run

Sealed run (PREREG_GEMINI_CLEAN.md): the plans carry no gold; the answers live in GOLD_JSONL and are joined
only when a row is written. agy runs under sandbox-exec with file reads denied in PLANS_DIR, the gold file's
directory and every --deny directory (agy's own --sandbox flag does not restrict reads). Before the first run a
canary file in a denied directory must be unreadable to the agent, or the run stops.

PLANS_DIR holds plans.jsonl and audio/ (run_dev --stage plans). The memory text is sent as the prompt with each
<|audio|> placeholder replaced by the clip's file name; the agent opens every clip with read_file (Gemini receives
the audio natively). The run is traced (stream-json): a run that calls any tool other than read_file, or does not
read every referenced clip, is invalid and retried once; a second invalid run is logged as an error.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

AGY = os.path.expanduser("~/.local/bin/agy")
INSTRUCT = ("You are a speech assistant answering from conversation memory. The memory below refers to audio clips "
            "by file name; each clip is a user turn. Open EVERY referenced clip with read_file, in the order given, "
            "and listen to it. Do not run any other tool, command or analysis. Then answer the final question, "
            "which is the last clip.\n\nFollow these instructions from the system prompt:\n{system}\n\n"
            "Memory and question:\n{user}\n\nEnd your reply with one line: ANSWER: <your answer>")


def prompt(row, audio_dir):
    parts = row["user"].split("<|audio|>")
    assert len(parts) == len(row["clips"]) + 1
    user = parts[0] + "".join(f"[audio clip: {os.path.join(audio_dir, c)}]" + p
                              for c, p in zip(row["clips"], parts[1:]))
    return INSTRUCT.format(system=row["system"], user=user)


def profile(deny, own=None):
    """sandbox-exec profile: no reads in the denied folders or in other runs' clip folders (q2g_* in the temp
    dir; the run's own folder is re-allowed), and no process may be started except agy itself, so the agent's
    shell tool cannot run anything (deviation 2026-10-28, PREREG_GEMINI_CLEAN.md §5)."""
    rules = " ".join(f'(subpath "{os.path.realpath(d)}")' for d in deny)
    tmp = re.escape(os.path.realpath(tempfile.gettempdir()))
    body = [f"(deny file-read* {rules})", f'(deny file-read* (regex #"^{tmp}/q2g_"))']
    if own:
        body.append(f'(allow file-read* (subpath "{os.path.realpath(own)}"))')
    body += ["(deny process-exec*)", f'(allow process-exec* (literal "{os.path.realpath(AGY)}"))']
    fd, p = tempfile.mkstemp(prefix="q2s_", suffix=".sb")
    with os.fdopen(fd, "w") as f:
        f.write("(version 1)\n(allow default)\n" + "\n".join(body) + "\n")
    return p


def canary(deny_dir, model, deny):
    """True if the agent can read neither a planted file in a denied directory nor one in another run's clip
    folder, by shell or read_file."""
    fd, secret = tempfile.mkstemp(prefix=".q2_canary_", suffix=".txt", dir=deny_dir)   # unique: shards run at once
    with os.fdopen(fd, "w") as f:
        f.write("CANARY-Q2-5581\n")
    ws, other = tempfile.mkdtemp(prefix="q2g_"), tempfile.mkdtemp(prefix="q2g_")
    peer = os.path.join(other, "clip.txt")
    open(peer, "w").write("CANARY-Q2-7702\n")
    sb = profile(deny, ws)
    try:
        cmd = ["sandbox-exec", "-f", sb, AGY, "--model", model, "--output-format", "stream-json",
               "--disable-slash-commands", "--add-dir", ws, "--print-timeout", "300s",
               "-p", f"Run the shell command `cat {secret} {peer}`, then read_file {secret} and read_file {peer}. "
                     "Reply with their exact contents."]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ws, timeout=400, stdin=subprocess.DEVNULL)
        return "CANARY-Q2-5581" not in r.stdout and "CANARY-Q2-7702" not in r.stdout
    finally:
        os.remove(secret)
        os.remove(sb)
        shutil.rmtree(ws, ignore_errors=True)
        shutil.rmtree(other, ignore_errors=True)


def run(row, src_dir, model, deny=None):
    # each run sees only its own referenced clips: a shared folder let runs open clips they were not given
    audio_dir = tempfile.mkdtemp(prefix="q2g_")
    for c in row["clips"]:
        shutil.copy(os.path.join(src_dir, c), audio_dir)
    sb = profile(deny, audio_dir) if deny is not None else None
    try:
        return _run(row, audio_dir, model, sb)
    finally:
        shutil.rmtree(audio_dir, ignore_errors=True)
        if sb:
            os.remove(sb)


def _run(row, audio_dir, model, sb=None):
    cmd = (["sandbox-exec", "-f", sb] if sb else []) + [AGY, "--model", model, "--output-format", "stream-json", "--disable-slash-commands",
           "--add-dir", audio_dir, "--print-timeout", "600s", "-p", prompt(row, audio_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=audio_dir, timeout=700, stdin=subprocess.DEVNULL)
    reads, other, answer, usage = [], [], None, {}
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        su = ev.get("step_update", {})
        if su.get("step_type") == "tool" and su.get("state") == "DONE":
            name = su.get("tool_name")
            path = (su.get("tool_info", {}).get("parameters") or {}).get("Path", "")
            failed = str(su.get("tool_info", {}).get("output", "")).lower().startswith("failed")
            if name == "read_file" and not (sb and failed):
                reads.append(os.path.basename(path))
            elif name != "read_file":
                other.append(name)
        if ev.get("event") == "result":
            answer = ev["result"].get("response")
            usage = ev["result"].get("usage", {})
    want = set(row["clips"])
    missing = sorted(want - set(reads))
    extra = sorted(set(reads) - want)
    # sealed runs: other tools are recorded but cannot start a process or read outside the run's own folder,
    # so only a missing clip or a successful read of another file invalidates the run
    ok = answer is not None and not missing and not extra and (sb is not None or not other)
    final = None
    if answer:
        lines = [l for l in answer.strip().splitlines() if l.strip().upper().startswith("ANSWER:")]
        final = lines[-1].split(":", 1)[1].strip() if lines else answer.strip().splitlines()[-1].strip()
    return dict(ok=ok, answer=final, raw=(answer or "")[-1500:], n_reads=len(reads), n_clips=len(row["clips"]),
                missing=missing[:5], extra=extra[:5], other_tools=other[:5], input_tokens=usage.get("input_tokens"),
                rc=r.returncode, stderr=r.stderr[-300:] if not ok else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans"); ap.add_argument("out"); ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--gold"); ap.add_argument("--deny", action="append", default=[])
    a = ap.parse_args()
    audio = os.path.abspath(os.path.join(a.plans, "audio"))
    deny, gold = None, {}
    if a.gold:
        gold = {r["qid"]: r["gold"] for r in map(json.loads, open(a.gold))}
        deny = [a.plans, os.path.dirname(os.path.abspath(a.gold))] + a.deny
        if not canary(deny[1], a.model, deny):
            sys.exit("canary readable inside the sandbox: not running")
        print("canary blocked; deny:", deny, flush=True)
    done = set()
    if os.path.exists(a.out):
        done = {(r["qid"], r["arm"]) for r in map(json.loads, open(a.out))}
    for line in open(os.path.join(a.plans, "plans.jsonl")):
        row = json.loads(line)
        if (row["qid"], row["arm"]) in done:
            continue
        res = run(row, audio, a.model, deny)
        if not res["ok"]:
            res = dict(run(row, audio, a.model, deny), attempt=2)
        if "RESOURCE_EXHAUSTED" in res["stderr"]:   # quota, not a reader failure: stop, row stays to do
            sys.exit(f"{row['qid']} {row['arm']}: model quota exhausted; resume after reset")
        row =dict(row, gold=gold[row["qid"]]) if a.gold else row
        rec = {k: row[k] for k in ("qid", "arm", "evidence_type", "answer_type", "gold", "total", "memory", "audio")}
        rec.update(model=a.model, **res)
        if not res["ok"]:
            rec["error"] = (f"invalid run: missing={res['missing']} extra={res['extra']} "
                            f"other={res['other_tools']} rc={res['rc']}")
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(row["qid"], row["arm"], "ok" if res["ok"] else rec["error"], flush=True)


if __name__ == "__main__":
    main()
