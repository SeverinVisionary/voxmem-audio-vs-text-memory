"""Open audio reader (PREREG_OMNI.md): Qwen3-Omni-30B-A3B-Instruct on the sealed clean-rerun plans. GPU. Resumable.

    python omni_reader.py PLANS_DIR OUT_JSONL [--limit N]       # Colab: generation, no gold anywhere
    python omni_reader.py --join ARMS_JSONL GOLD_JSONL OUT_JSONL # local: add the gold for scoring

PLANS_DIR holds plans.jsonl and audio/ (run_dev --stage plans --plan-subset all; no gold). Each row becomes the
system turn plus one user turn in which the memory text and the WAV clips alternate where the plan has
<|audio|>, the reader's native interleaved layout. Thinker only (no speech output), greedy, max_new from the
Phase 1 frozen settings. A row that fails (e.g. out of memory) is logged with an error and retried on the next
start; one still failing after the rerun stage is scored wrong.
"""
import argparse
import json
import os
import time

M = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_minicpm.json")))["settings"]["max_new"]


def messages(row, audio_dir):
    parts = row["user"].split("<|audio|>")
    assert len(parts) == len(row["clips"]) + 1
    content = []
    for i, p in enumerate(parts):
        if p.strip():
            content.append({"type": "text", "text": p})
        if i < len(row["clips"]):
            content.append({"type": "audio", "audio": os.path.join(audio_dir, row["clips"][i])})
    return [{"role": "system", "content": [{"type": "text", "text": row["system"]}]},
            {"role": "user", "content": content}]


class Omni:
    def __init__(self):
        import torch
        import transformers
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        self.torch = torch
        t0 = time.time()
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            M, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        self.model.disable_talker()
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(M)
        self.info = dict(model=M, transformers=transformers.__version__, torch=torch.__version__, attn="sdpa",
                         load_s=round(time.time() - t0, 1))

    def generate(self, msgs):
        from qwen_omni_utils import process_mm_info
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)
        x = self.proc(text=text, audio=audios, return_tensors="pt", padding=True, use_audio_in_video=False)
        x = x.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            o, _ = self.model.generate(**x, thinker_max_new_tokens=MAX_NEW, thinker_do_sample=False, return_audio=False,
                                       use_audio_in_video=False)
        torch.cuda.synchronize()
        n_in = x["input_ids"].shape[1]
        new = o[0][n_in:]
        return dict(answer=self.proc.batch_decode(new[None], skip_special_tokens=True)[0].strip(), n_new=int(len(new)),
                    hit_max=bool(len(new) >= MAX_NEW), seconds=round(time.perf_counter() - t, 3), n_in=int(n_in),
                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def join(arms, gold, out):
    g = {r["qid"]: r["gold"] for r in map(json.loads, open(gold))}
    with open(out, "w") as f:
        for r in map(json.loads, open(arms)):
            f.write(json.dumps(dict(r, gold=g[r["qid"]])) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--limit", type=int); ap.add_argument("--join", action="store_true")
    a = ap.parse_args()
    if a.join:
        return join(*a.paths)
    plans, out = a.paths
    audio = os.path.join(plans, "audio")
    rows = [json.loads(l) for l in open(os.path.join(plans, "plans.jsonl"))]
    assert not any("gold" in r for r in rows), "gold in plans"
    done = set()
    if os.path.exists(out):
        done = {(r["qid"], r["arm"]) for r in map(json.loads, open(out)) if "error" not in r}
    todo = [r for r in rows if (r["qid"], r["arm"]) not in done][:a.limit]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    rd = Omni()
    print(json.dumps(rd.info), flush=True)
    streak = 0
    for row in todo:
        rec = {k: row[k] for k in ("qid", "arm", "evidence_type", "answer_type", "total", "memory", "audio")}
        rec.update(model=M, n_clips=len(row["clips"]))
        try:
            rec.update(rd.generate(messages(row, audio)))
        except Exception as e:  # noqa: BLE001  (OOM): logged, retried on the next start
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rd.torch.cuda.empty_cache()
        streak = streak + 1 if "error" in rec else 0
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(row["qid"], row["arm"], rec.get("seconds"), rec.get("n_in"), rec.get("error", "")[:80], flush=True)
        if streak >= 5:
            raise SystemExit("5 failures in a row: stopping (a broken setup, not a row problem)")


if __name__ == "__main__":
    main()
