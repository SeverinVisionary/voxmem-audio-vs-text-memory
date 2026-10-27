"""CPU-only checks of memory.py with a fake tokenizer and fake features (no models, no downloads).

    python test_memory.py
"""
import copy

import numpy as np

import memory as M
from memory import Builder, Counter, ItemMemory, audio_tokens, minicpm_audio_tokens


class Tok:
    """Whitespace tokenizer with a Llama-like chat template."""

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": s.split()}

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "".join(f"<hdr> {m['role']} </hdr> {m['content']} <eot> " for m in msgs) + "<hdr> assistant </hdr> "


class FakeFeats:
    def __init__(self, keys, rng):
        self.f, self.selftags, self._e = {}, {}, {}
        for i, k in enumerate(keys):
            spk = np.eye(4)[i % 2] + 0.01 * rng.standard_normal(4)
            self._e[k] = (spk / np.linalg.norm(spk), rng.standard_normal(6))
            self.f[k] = dict(key=k, sv=dict(text=f"words of turn {i} here", emo="NEUTRAL", event="Speech", lang="en"),
                             e2v=dict(labels=["neutral", "sad", "happy"], scores=[0.6, 0.3, 0.1]),
                             ast=dict(top=[["Speech", 0.9], ["Dog", 0.3]], bg=[["Dog", 0.3]]),
                             pros=dict(dur=3.0, f0_med=150 + i, voiced=0.6, rms_db=-25.0 + i, rate_wps=2.5))
            self.selftags[k] = '{"emotion":"neutral","pitch":"mid"}'

    def emb(self, k):
        e = self._e[k]
        return e[0], e[1] / np.linalg.norm(e[1])


def make_item(qid, n_sess=3, per=4, tag=""):
    sessions = []
    for s in range(n_sess):
        sid = f"s{tag}{s}"
        turns = []
        for t in range(per):
            turns.append(dict(role="user", key=f"{sid}#{2 * t}"))
            turns.append(dict(role="assistant", key=f"{sid}#{2 * t + 1}", text="sure thing"))
        sessions.append(dict(session_id=sid, timestamp=f"2025-01-0{s + 1}", turns=turns))
    return dict(question_id=qid, query_timestamp="2025-02-01", sessions=sessions)


LENS = {"__question__": 16000 * 2}


def setup(qid="q1", tag="", **kw):
    rng = np.random.default_rng(0)
    item = make_item(qid, tag=tag)
    keys = [t["key"] for s in item["sessions"] for t in s["turns"] if t["role"] == "user"]
    F = FakeFeats(keys, rng)
    LENS.update({k: 16000 * (2 + i % 3) for i, k in enumerate(keys)})
    mem = ItemMemory(item, F)
    return Builder(item, mem, Counter(Tok()), LENS.__getitem__, **kw), item, F, LENS


def main():
    M.system_prompt = lambda answer_line=0: "You answer questions ."     # avoid reading repo prompt file
    assert audio_tokens(16000) == 7 and audio_tokens(480000) == 188 and audio_tokens(496000) == 195
    # MiniCPM-o 4.5 clip positions measured on dev (out_dev/minicpm_positions.json; each includes one "\n" token)
    for sec, n in [(5.84, 61), (5.4, 57), (3.08, 33), (3.0, 33), (1.76, 20), (7.32, 76)]:
        assert minicpm_audio_tokens(round(sec * 16000)) + 1 == n, (sec, n)
    b, item, F, lens = setup()
    for f in (0.10, 0.25, 0.50):
        tgt = b.target(f)
        for arm in ("T", "G", "E"):
            p = b.arm(arm, f)
            n = b.positions(p)
            assert tgt - 1 <= n <= tgt, (arm, f, n, tgt)
            user, clips = b.render(p)
            assert user.count("<|audio|>") == len(clips)
            assert all(k in b.mem.user for k in p["audio"])
    # G uses audio, T does not; F has all audio; A has no records
    assert not b.arm("T", 0.25)["audio"] and b.arm("G", 0.5)["audio"]
    assert len(b.arm("F")["audio"]) == len(b.mem.user) and not b.arm("A")["rec"]
    assert b.positions(b.arm("F")) > b.target(0.5)
    # E keeps the most recent turns
    e = list(b.arm("E", 0.25)["audio"])
    assert e == b.mem.user[-len(e):]
    # causality: records of early turns do not change when later turns are removed
    full = {k: b.mem.record(k, "Dp", "kv") for k in b.mem.user}
    cut = copy.deepcopy(item); cut["sessions"] = cut["sessions"][:2]
    early = ItemMemory(cut, F)
    assert all(early.record(k, "Dp", "kv") == full[k] for k in early.user)
    # online speaker ids alternate as constructed
    assert len(set(b.mem.spk.values())) == 2
    # mismatched arm uses donor audio in own slots
    d, *_ = setup("q2", tag="d")
    gmm = b.arm("Gmm", 0.5, donor=[d])
    g = b.arm("G", 0.5)
    assert gmm and all(src.startswith("sd") for src in gmm["audio"].values())
    assert list(gmm["audio"]) == list(g["audio"]), "G_mismatched must use exactly G's slots"
    # online G: the kept set after turn t depends only on turns <= t (prefix consistency of the eviction log)
    from memory import select_reservoir
    keys, cost = b.mem.user, b.sel_cost
    caps = b.caps(0.5, keys, cost, b.target(0.5) - b.positions(b.plan_m()))
    for j in range(1, len(keys)):
        pre = select_reservoir(keys[:j], cost, caps[:j])
        full = select_reservoir(keys, cost, caps)
        assert set(full) & set(keys[:j]) <= set(pre), "a turn discarded at time j came back later"
    # shuffled records are a permutation
    sh = b.arm("shuf")
    assert sorted(sh["rec"].values()) == sorted(b.plan_m()["rec"].values())
    # every writer and format renders
    for w in M.WRITERS:
        for fm in M.FORMATS:
            assert b.mem.record(b.mem.user[3], w, fm)
    led = b.ledger(b.arm("G", 0.25))
    assert led["memory"] == led["audio"] + led["text"]
    # the chat layout carries the same records, audio and order as the flat one
    for plan in (b.arm("F"), b.arm("T", 0.25), b.arm("N"), b.arm("A")):
        user, clips = b.render(plan)
        msgs, cclips = b.render_chat(plan)
        assert cclips == clips and sum(m["content"].count("<|audio|>") for m in msgs) == len(clips)
        assert all(v in "\n".join(m["content"] for m in msgs) for v in plan["rec"].values())
        assert msgs[-1]["content"].endswith("<|audio|>") and msgs[-1]["role"] == "user"
    print("memory tests OK")


if __name__ == "__main__":
    main()
