"""Source clustering for the resampling units (PREREGISTRATION §7). Audio and metadata only; no outcomes.

    python clusters.py DATA FEAT OUT_JSON [--esc DIR]

Voices: every user turn of every family's evidence sessions, WavLM-SV embedding (features.py), average-linkage
cosine clustering into the 30 VCTK source voices (reference-free; the VCTK 0.92 originals are an 11 GB archive).
Voice purity = share of evidence-session turns in their session's majority voice.
Background clips (environmental families): normalised cross-correlation of each evidence-session user turn with
the 45 ESC-50 originals; a turn matches its best clip when that score is >= 3x the second best. Clip-match
accuracy is checked on the families whose gold answer is a single clip category.
Components: families sharing any source (session-majority voices, matched clips) are merged. If the largest
component holds more than 25% of the families, the units fall back to one source per family: the voice of most
evidence turns (speaker, paralinguistic) or the clip matched most often (environmental; voice if none matched).
"""
import argparse
import collections
import glob
import json
import os
import zipfile

import numpy as np

from common import Q2, load_eval, load_item
from data import load_audio
from memory import Feats

N_VOICES = 30
GIANT = 0.25
ESC_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def evidence_turns(data, q):
    ev, it = load_eval(data, q), load_item(data, q)
    return [(s["session_id"], t["key"]) for s in it["sessions"] if s["session_id"] in ev["evidence_session_ids"]
            for t in s["turns"] if t["role"] == "user"], ev


def esc_clips(esc_dir):
    """The 45 VoxMem background clips at 16 kHz, from the ESC-50 archive (downloaded once into esc_dir)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    meta = json.load(open(Q2 / "phase0/out/vox_metadata__environmental_sources.json"))["clips"]
    z = os.path.join(esc_dir, "esc50.zip")
    if not os.path.exists(z):
        import urllib.request
        os.makedirs(esc_dir, exist_ok=True)
        urllib.request.urlretrieve(ESC_URL, z)
    out = {}
    with zipfile.ZipFile(z) as f:
        names = {os.path.basename(n): n for n in f.namelist() if n.endswith(".wav")}
        for c in meta:
            wav, sr = sf.read(f.open(names[c["esc50_clip"]]), dtype="float32")
            wav = wav.mean(1) if wav.ndim > 1 else wav
            out[c["category"]] = resample_poly(wav, 160, sr // 100).astype(np.float32) if sr != 16000 else wav
    return out


def ncc_max(x, c):
    """Max over lags of the correlation coefficient between x and every equally long window of c (c padded)."""
    n = len(x)
    if len(c) < n:
        c = np.concatenate([c, np.zeros(n - len(c), np.float32)])
    L = 1 << int(np.ceil(np.log2(len(c) + n)))
    num = np.fft.irfft(np.fft.rfft(c, L) * np.conj(np.fft.rfft(x, L)), L)[:len(c) - n + 1]
    e = np.concatenate([[0.0], np.cumsum(c.astype(np.float64) ** 2)])
    win = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-12))
    return float(np.max(np.abs(num) / (win * (np.linalg.norm(x) + 1e-12))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("feat"); ap.add_argument("out"); ap.add_argument("--esc", default="/content/esc")
    a = ap.parse_args()
    from sklearn.cluster import AgglomerativeClustering
    qids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.data, "items", "*.json")))
    F = Feats(a.feat)
    turns, evs = {}, {}
    for q in qids:
        turns[q], evs[q] = evidence_turns(a.data, q)
    keys = sorted({k for q in qids for _, k in turns[q]})
    X = np.stack([F.emb(k)[0] for k in keys])
    lab = AgglomerativeClustering(n_clusters=N_VOICES, metric="cosine", linkage="average").fit_predict(X)
    voice = dict(zip(keys, (int(x) for x in lab)))
    sess = collections.defaultdict(list)
    for q in qids:
        for s, k in turns[q]:
            sess[s].append(voice[k])
    purity = sum(collections.Counter(v).most_common(1)[0][1] for v in sess.values()) / sum(map(len, sess.values()))
    sources, single = {}, {}
    for q in qids:
        vs = {collections.Counter(sess[s]).most_common(1)[0][0] for s, _ in turns[q]}
        sources[q] = {f"v{v}" for v in vs}
        single[q] = f"v{collections.Counter(voice[k] for _, k in turns[q]).most_common(1)[0][0]}"
    clips = esc_clips(a.esc)
    names = sorted(clips)
    match_check, clip_of = [], {}
    for q in qids:
        if evs[q]["evidence_type"] != "environmental_sound":
            continue
        got = collections.Counter()
        for _, k in turns[q]:
            x = load_audio(a.data, k)
            sc = sorted(((ncc_max(x, clips[n]), n) for n in names), reverse=True)
            if sc[0][0] >= 3 * sc[1][0]:
                got[sc[0][1]] += 1
        sources[q] |= {f"c:{n}" for n in got}
        if got:
            clip_of[q] = got.most_common(1)[0][0]
            single[q] = f"c:{clip_of[q]}"
        g = json.loads(evs[q]["gold_json"])
        if isinstance(g, str) and g in clips:
            match_check.append(int(clip_of.get(q) == g))
    parent = {q: q for q in qids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    owner = {}
    for q in qids:
        for s in sources[q]:
            if s in owner:
                parent[find(q)] = find(owner[s])
            else:
                owner[s] = q
    comp = {q: find(q) for q in qids}
    sizes = collections.Counter(comp.values())
    giant = max(sizes.values()) / len(qids)
    fallback = giant > GIANT
    final = single if fallback else comp
    res = dict(component=final, n_families=len(qids), n_units=len(set(final.values())),
               voice_purity=round(purity, 4), voice_purity_ok=purity >= 0.9,
               clip_match_accuracy=round(float(np.mean(match_check)), 4) if match_check else None,
               clip_match_n=len(match_check), env_with_clip=len(clip_of),
               components_all_sources=dict(n=len(sizes), largest_share=round(giant, 4)),
               fallback_one_source_per_family=fallback,
               unit_sizes=sorted(collections.Counter(final.values()).values(), reverse=True))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "component"}, indent=1))


if __name__ == "__main__":
    main()
