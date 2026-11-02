# Q2 — positive control: can the open reader use old clips at all? (preregistration, FROZEN 2026-11-02)

Drafted and frozen 2026-11-02, after v3.1, before any answer for the arms below existed. Plans, notebook
builder and analysis script are committed with this file.

## 1. Question

v3.1 found no gain from old audio over a transcript for MiniCPM-o or Qwen3-Omni, but no contrast showed either
reader uses old clips at all. So "audio adds nothing" cannot be told apart from "this reader cannot use old
audio". MiniCPM-o already failed the dev oracle (transcript + evidence-session clips, −3.1 pp vs T\* on 32
families, `DEV_RESULTS.md`). Qwen3-Omni has never been given the right clips. This study asks, for Qwen3-Omni:

1. With only the evidence sessions in context, does audio beat their transcript? (perception of old clips)
2. Inside the full transcript, does adding the evidence sessions' clips beat the transcript alone? (use of the
   right clips amid history)

## 2. Design

- Reader, decoding, harness, judges, matcher: exactly as `PREREG_OMNI.md` §2 (Qwen3-Omni-30B-A3B-Instruct,
  thinker only, bf16, greedy, `max_new` 48; judge 1 `claude-opus-5-5`, judge 2 `claude-sonnet-5-5` on a seeded
  10% plus every judge-1/matcher disagreement).
- Families: the same 142 holdout families. Units: the holdout dominant-source clusters.
- New arms (426 rows), derived from the sealed clean-rerun plans by `pilot/make_posctrl_plans.py` (no gold;
  release asset `q2-posctrl-plans-v1/plans_posctrl.zip`, sha256 ea255dc2…):
  - **Ev-A**: only the evidence sessions, user turns as native audio, assistant turns as text, in order.
  - **Ev-M**: the same sessions, user turns as their ASR transcript (M:asr restricted to evidence sessions).
  - **O**: the full M:asr transcript plus the evidence sessions' user clips in their own slots (the existing
    `oracle` arm of `memory.py`), over budget by design.
- Reused, not re-run: N, A and M:asr from the v3 run (`out_omni/arms_omni.jsonl`, frozen dc3a14f0b). Same
  reader, plans and greedy decoding; the notebook first re-runs 20 M:asr rows (hash order)
  and runs the new arms only if at least 18 of 20 answers equal v3's (case-insensitive). Otherwise the run
  stops, and a re-run of N, A and M:asr with the new arms is decided and logged in §6 before any new-arm
  answer is seen.
- Evidence-session ids come from VoxMem's item metadata. Plans for Ev-A/Ev-M/O therefore reveal which sessions
  hold the answer, but not the answer. This is a diagnostic about the reader, not a memory policy.

## 3. Analysis

`pilot/analyze_posctrl.py`. Type-stratified cluster bootstrap-t with the holdout dominant-source units,
one-sided sign-flip p, seeds from 3001, as in `analyze_test`. The unstratified bound is reported next to P1
and P2. Judging: the new rows go through `judge.py` on their own; N, A and M:asr keep their v3 verdicts.

- **P1 (primary). Ev-A − Ev-M > 0**, one-sided α = 0.025. Lower bound above 0 → HEARS-OLD-CLIPS; otherwise
  NOT-SHOWN.
- **P2 (fixed sequence after P1). O − M:asr > 0**, tested only if P1 passes, so the pair keeps α = 0.025.
  Lower bound above 0 → USES-RIGHT-CLIPS; otherwise NOT-SHOWN. If P1 fails, P2 is reported as an estimate.
- Estimates: (Ev-A − Ev-M) − (A − M:asr), the cost of history length; P1 and P2 by evidence type
  (environmental, paralinguistic, speaker); Ev-A − N and Ev-M − N.
- Power: from v3 the paired SE on 142 families is about 3.0–3.5 pp, so the design detects about 9–10 pp at 80%.
  VoxMem's Gemini check (46.2% with audio vs 4.4% with the synthesis script on acoustic items) is far larger;
  an effect under 9 pp is reported as an estimate, not a pass.

## 4. What follows from it

| P1 | P2 | Reading for the paper |
|---|---|---|
| NOT-SHOWN | not tested | Instrument null: Qwen3-Omni does not use old clips even when isolated, so v3.1's result says nothing about the value of old audio, only about these readers. |
| HEARS | NOT-SHOWN | The reader hears old clips alone but not amid history: the transcript-suffices result reflects use of context, not the information in audio. |
| HEARS | USES | The reader can use the right clips; A ≈ M:asr then means the full history dilutes them. Selection of clips is the open problem. |

## 5. Compute

One Colab run, run-once guarded, same notebook skeleton as `q2_omni.ipynb`: 426 new rows plus 20 replay rows.
The v3 run took about 16 min of generation for 710 rows plus 3.5 min download on a G4 (~10.4 CU/h, ~4 CU).
Ev-A/Ev-M are much shorter than A; O is M:asr plus a few clips. Estimate 10–14 min total, **about 2–3 CU**,
on the cheapest card with more than 75 GB. Judging: ~450 judge-1 calls plus ~60 judge-2 calls.

## 6. Deviations

- 2026-11-02, after the answers were judged: `analyze_posctrl.py` was given the gold-joined file of the new
  arms (`out_posctrl/arms_posctrl_gold.jsonl`, the 426 rows that were judged) instead of the raw run file,
  because the 20 replay rows are not judged and `analyze_test.load` needs a verdict for every row. The replay
  rows play no part in the analysis; the code is unchanged.
