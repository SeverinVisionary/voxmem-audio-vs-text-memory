"""Positive-control plans (PREREG_POSCTRL.md §2), derived from the sealed clean-rerun plans. CPU, no gold.

    python make_posctrl_plans.py PLANS_DIR ITEMS_CSV OUT_DIR    # PLANS_DIR: unzipped plans_clean.zip

Every family's A and M:asr plans render the same sessions and turns (A: `User: <|audio|>`, M:asr: `User: said=…`).
A session block is mapped to its session id through A's clips (`sess_XXXXXX_<turn>.wav`); the evidence sessions
come from VoxMem's item metadata. Arms:
  Ev-A   A restricted to the evidence sessions (their user clips and assistant text)
  Ev-M   M:asr restricted to the evidence sessions
  O      the full M:asr plan, with each evidence-session user turn's clip after its transcript (memory.py `oracle`)
  M:asr#replay   the v3 M:asr plan unchanged, on 20 families in hash order (determinism check against v3)
Replay rows come first. OUT_DIR gets plans.jsonl and audio/ with only the clips these rows use.
"""
import csv
import json
import os
import re
import shutil
import sys

from common import hkey

HEAD = "Conversation history:\n"
SESSION = re.compile(r"^\[Session [^\]]+\]$")
N_REPLAY = 20


def blocks(user):
    """(preamble, [session blocks as line lists], tail) of a rendered plan with history."""
    body, tail = user.rsplit("\n[Now ", 1)
    assert body.startswith(HEAD)
    out = []
    for line in body[len(HEAD):].split("\n"):
        if SESSION.match(line):
            out.append([line])
        else:
            out[-1].append(line)
    return out, "\n[Now " + tail


def render(bl, tail):
    return HEAD + "\n".join(l for b in bl for l in b) + tail


def derive(a, m, ev):
    """Ev-A, Ev-M and O user texts and clip lists for one family."""
    ba, tail = blocks(a["user"])
    bm, tail_m = blocks(m["user"])
    assert tail == tail_m and len(ba) == len(bm)
    clips = iter(a["clips"][:-1])
    sid, a_clips = [], []
    for b, c in zip(ba, bm):
        assert len(b) == len(c) and all(x.startswith("User:") == y.startswith("User:") for x, y in zip(b, c))
        cs = [next(clips) for x in b if x == "User: <|audio|>"]
        assert len(cs) == sum(x.startswith("User:") for x in b)
        ids = {x.rsplit("_", 1)[0] for x in cs}
        assert len(ids) == 1, ids
        sid.append(ids.pop()); a_clips.append(cs)
    assert next(clips, None) is None
    keep = [i for i, s in enumerate(sid) if s in ev]
    assert len(keep) == len(ev), (sid, ev)
    q = a["clips"][-1:]
    o_blocks = [[x + " <|audio|>" if i in keep and x.startswith("User:") else x for x in c] for i, c in enumerate(bm)]
    return {"Ev-A": (render([ba[i] for i in keep], tail), [x for i in keep for x in a_clips[i]] + q),
            "Ev-M": (render([bm[i] for i in keep], tail), q),
            "O": (render(o_blocks, tail), [x for i in keep for x in a_clips[i]] + q)}


def main():
    plans, items_csv, out = sys.argv[1:]
    rows = {(r["qid"], r["arm"]): r for r in map(json.loads, open(os.path.join(plans, "plans.jsonl")))}
    assert not any("gold" in r for r in rows.values()), "gold in plans"
    ev = {r["question_id"]: set(json.loads(r["evidence_session_ids"])) for r in csv.DictReader(open(items_csv))}
    qids = sorted({q for q, _ in rows})
    replay = sorted(qids, key=lambda q: hkey("posctrl-replay", q))[:N_REPLAY]
    keep = ("qid", "system", "evidence_type", "answer_type")
    new = [dict(rows[(q, "M:asr")], arm="M:asr#replay") for q in replay]
    for q in qids:
        for arm, (user, clips) in derive(rows[(q, "A")], rows[(q, "M:asr")], ev[q]).items():
            assert user.count("<|audio|>") == len(clips)
            new.append(dict({k: rows[(q, "A")][k] for k in keep}, arm=arm, user=user, clips=clips,
                            total=None, memory=None, audio=None, n_audio_turns=len(clips) - 1))
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    with open(os.path.join(out, "plans.jsonl"), "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
            for c in r["clips"]:
                dst = os.path.join(out, "audio", c)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(plans, "audio", c), dst)
    print(len(new), "rows,", len(os.listdir(os.path.join(out, "audio"))), "clips,", len(qids), "families")


if __name__ == "__main__":
    main()
