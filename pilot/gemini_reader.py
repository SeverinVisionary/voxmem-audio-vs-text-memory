"""Strong-reader ceiling (PREREG_HOLDOUT §4): Gemini through the `agy` CLI on exported plans. Resumable.

    python gemini_reader.py PLANS_DIR OUT_JSONL [--model gemini-3.8-flash-high]

PLANS_DIR holds plans.jsonl and audio/ (run_dev --stage plans). The memory text is sent as the prompt with each
<|audio|> placeholder replaced by the clip's file name; the agent opens every clip with read_file (Gemini receives
the audio natively). The run is traced (stream-json): a run that calls any tool other than read_file, or does not
read every referenced clip, is invalid and retried once; a second invalid run is logged as an error.
"""
import argparse
import json
import os
import shutil
import subprocess
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


def run(row, src_dir, model):
    # each run sees only its own referenced clips: a shared folder let runs open clips they were not given
    audio_dir = tempfile.mkdtemp(prefix="q2g_")
    for c in row["clips"]:
        shutil.copy(os.path.join(src_dir, c), audio_dir)
    try:
        return _run(row, audio_dir, model)
    finally:
        shutil.rmtree(audio_dir, ignore_errors=True)


def _run(row, audio_dir, model):
    cmd = [AGY, "--model", model, "--output-format", "stream-json", "--disable-slash-commands",
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
            (reads if name == "read_file" else other).append(os.path.basename(path) if name == "read_file" else name)
        if ev.get("event") == "result":
            answer = ev["result"].get("response")
            usage = ev["result"].get("usage", {})
    want = set(row["clips"])
    missing = sorted(want - set(reads))
    extra = sorted(set(reads) - want)
    ok = answer is not None and not other and not missing and not extra
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
    a = ap.parse_args()
    audio = os.path.abspath(os.path.join(a.plans, "audio"))
    done = set()
    if os.path.exists(a.out):
        done = {(r["qid"], r["arm"]) for r in map(json.loads, open(a.out))}
    for line in open(os.path.join(a.plans, "plans.jsonl")):
        row = json.loads(line)
        if (row["qid"], row["arm"]) in done:
            continue
        res = run(row, audio, a.model)
        if not res["ok"]:
            res = dict(run(row, audio, a.model), attempt=2)
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
