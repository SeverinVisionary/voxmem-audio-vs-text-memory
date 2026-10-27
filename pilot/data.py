"""Extract VoxMem 8K items to a local directory, splitting writer-visible and analysis-only fields.

    python data.py OUT_DIR dev            # all dev families
    python data.py OUT_DIR dev 2          # first 2 dev families (CPU smoke)
    python data.py OUT_DIR dev 0/2        # shard 0 of 2

Writes OUT_DIR/items/<qid>.json (writer-visible), OUT_DIR/eval/<qid>.json (analysis-only),
OUT_DIR/audio/<session_id>.npz (int16 16 kHz user-turn audio, keys t<turn_idx>), OUT_DIR/qaudio/<qid>.npy.
"""
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import SR, VOX, items_csv, split_ids


def to16k(a):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype("float32")
    return wav


def i16(wav):
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def extract(out, qids):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    meta = items_csv()
    for d in ("items", "eval", "audio", "qaudio"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    by_file = defaultdict(set)
    for q in qids:
        by_file[meta[q]["source_file"]].add(q)
    done = set()
    for src, want in sorted(by_file.items()):
        pf = pq.ParquetFile(hf_hub_download(VOX, src, repo_type="dataset"))
        for rg in range(pf.num_row_groups):
            ids = pf.read_row_group(rg, columns=["question_id"]).column(0).to_pylist()
            if not want & set(ids):
                continue
            for row in pf.read_row_group(rg).to_pylist():
                q = row["question_id"]
                if q not in want or q in done:
                    continue
                write_item(out, row)
                done.add(q)
    missing = set(qids) - done
    assert not missing, f"not found: {sorted(missing)}"
    return sorted(done)


def write_item(out, row):
    q = row["question_id"]
    sessions, roles, synth = [], {}, {}
    for s in row["sessions"]:
        sid = s["session_id"]
        roles[sid] = s.get("role")
        turns, clips = [], {}
        for ti, t in enumerate(s["turns"]):
            key = f"{sid}#{ti}"
            if t["role"] == "user":
                assert (t.get("audio") or {}).get("bytes"), f"missing user audio {q} {key}"
                turns.append(dict(role="user", key=key))
                clips[f"t{ti}"] = i16(to16k(t["audio"]))
                synth[key] = t.get("text")
            else:
                turns.append(dict(role="assistant", key=key, text=t.get("text") or ""))
        sessions.append(dict(session_id=sid, timestamp=s["timestamp"], turns=turns))
        path = os.path.join(out, "audio", f"{sid}.npz")
        if not os.path.exists(path):                # sessions are shared across items; content is identical
            np.savez_compressed(path, **clips)
    np.save(os.path.join(out, "qaudio", f"{q}.npy"), i16(to16k(row["question_audio"])))
    item = dict(question_id=q, query_timestamp=row["query_timestamp"], sessions=sessions)
    ev = dict(question_id=q, family_id=row["family_id"], evidence_type=row["evidence_type"],
              memory_operation=row["memory_operation"], subtype=row["subtype"], question_text=row["question_text"],
              gold_json=row["gold_json"], answer_type=row["answer_type"],
              answer_normalization=row["answer_normalization"], evidence_session_ids=row["evidence_session_ids"],
              session_roles=roles, user_synthesis_text=synth)
    json.dump(item, open(os.path.join(out, "items", f"{q}.json"), "w"))
    json.dump(ev, open(os.path.join(out, "eval", f"{q}.json"), "w"))


def load_audio(data_dir, key):
    sid, ti = key.split("#")
    with np.load(os.path.join(data_dir, "audio", f"{sid}.npz")) as z:
        return z[f"t{ti}"].astype(np.float32) / 32767


if __name__ == "__main__":
    out, split = sys.argv[1], sys.argv[2]
    qids = split_ids(split)
    if len(sys.argv) > 3:                      # "N" = first N items; "i/n" = shard i of n
        sel = sys.argv[3]
        if "/" in sel:
            i, n = map(int, sel.split("/"))
            qids = qids[i::n]
        else:
            qids = qids[: int(sel)]
    print(len(extract(out, qids)), "items")
