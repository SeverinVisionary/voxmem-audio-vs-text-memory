# Q2 Phase 1b preregistration — holdout study (v1, 2026-10-26)

Status: **frozen** by the commit that adds this file (SHA recorded on #1082) before any holdout family is run.
Phase 1 (`PREREGISTRATION.md` v3, test split) ended NO-CUE, so Phase 2 does not run. This is a new study
with new questions, not a rescue of Phase 1: its outcomes do not change the Phase 1 decision.

## 1. Questions

Phase 1 found that, for MiniCPM-o 4.5, a compact text memory matches full audio and old audio adds nothing at
matched positions. Phase 1b asks why, on fresh families:

- **B1 (primary): does the benchmark need audio memory for this reader?** Transcript-only memory (`M:asr`,
  the writer's own SenseVoice transcript, no acoustic fields) versus the full user-audio history (`A`, the
  VoxMem setting). VoxMem reports that a transcript does not answer its acoustic items (Gemini 3.7 Flash:
  4.4% vs 46.2% with audio). Hypothesis: for MiniCPM-o, M:asr is **non-inferior** to A at margin 10 pp.
- **B2 (secondary): which text field carries the cue?** Per evidence type, transcript plus the matching
  field group versus transcript only: speaker → speaker labels (`M:D-spk`), paralinguistic → emotion labels
  (`M:D-emo`), environmental → sound-event/background labels (`M:D-env`). One-sided > 0, Holm over the three.
  Also reported: full record `M:D` − `M:asr`.
- **B3 (secondary, estimates only): does the Phase 1 result replicate?** ΔF = F − T\*@0.25 and
  ΔG = G@0.25 − T\*@0.25 with 95% intervals.
- **C (ceiling, preregistered but exploratory): is the null reader-limited?** A strong closed reader
  (Gemini 3.8 Flash, high) on a 48-family subset, arms N, A, M:asr, T\*@0.25, G@0.25, F, rendered exactly as
  for MiniCPM-o (same text, same clips; budgets in MiniCPM-o positions). Reported: A − M:asr, F − T\*,
  G − T\*, A − N for both readers on those families, and the difference in differences
  (A − M:asr)_Gemini − (A − M:asr)_MiniCPM.

## 2. Data, reader, settings

- The 142 holdout families (`phase0/splits_8k.csv`, 38 speaker / 67 paralinguistic / 37 environmental),
  never run before. Ceiling subset: the first 16 families per type in `hkey("ceiling", qid)` order (48).
- MiniCPM-o 4.5, flat layout, greedy, all settings from `pilot/frozen_minicpm.json` (Phase 1 frozen: writer D,
  key=value, prosody extra queue, hash-priority G, 48 new tokens, answer line 0). Nothing is re-tuned.
- Field-group writers: `D-spk` = transcript + session-local speaker label; `D-emo` = transcript +
  emotion2vec label/score + SenseVoice emotion; `D-env` = transcript + SenseVoice event + AST background labels.
- Gemini reader (`pilot/gemini_reader.py`): `agy` CLI, model `gemini-3.8-flash-high` (no temperature
  control). The memory text is the prompt and each clip is opened with `read_file`, so audio reaches the
  model natively but as tool results, not inline. Every run is traced: one that calls any other tool, or
  does not read every referenced clip, is invalid, retried once, then scored incorrect and logged.

## 3. Scoring and analysis

- Judging as Phase 1 (`pilot/judge.py`, blind, deterministic matcher in parallel, seeded 10% plus
  disagreements to a second judge). For Gemini's answers the judges are swapped (judge 1
  `claude-opus-5-5`, judge 2 `gemini-3.8-flash-high`) so that Gemini does not grade its own answers first.
- Resampling units from `pilot/clusters.py` on the holdout (same rule as Phase 1 §7, including the
  one-source fallback). Estimators as Phase 1: family-weighted paired differences, cluster sign-flip
  one-sided p (10,000 flips), type-stratified cluster bootstrap-t 95% intervals (10,000 resamples).
- **B1 decision:** lower 95% bound of M:asr − A > −10 pp → **TRANSCRIPT-SUFFICES**; upper bound < 0 →
  **AUDIO-NEEDED**; otherwise **INCONCLUSIVE**.
- Reader failures > 5% in A or M:asr make B1 INVALID. Analysis script `pilot/analyze_holdout.py`
  (committed with this file).
- Power: n = 142; with discordance ≈ 0.2 the 95% half-width is ≈ 7–9 pp, so B1 can pass at the 10 pp margin
  only if M:asr ≈ A. B2 per type (n = 37–67) is underpowered and is read as an estimate.

## 4. Run

One Colab L4 notebook (`pilot/q2_minicpm_holdout.ipynb`, run once): data, features, clusters, the 10 holdout
arms (N, A, M:asr, M:D-emo, M:D-env, M:D-spk, M:D, T\*@0.25, G@0.25, F) with one rerun of failed rows, then
the ceiling subset exported as prompts + WAV. The Gemini runs follow locally on the exported plans.

## 5. Deviations

Logged here with date and whether holdout outcomes had been seen.

- 2026-10-26, before any Gemini answer was judged (MiniCPM-o holdout results had been analysed): the first
  Gemini pass ran every family against one shared audio folder. 11 runs read clips they were not given
  (for example a transcript-only run opening 40 memory clips) and 6 transcript-only runs used shell commands.
  The traced-run rule is extended: a run that reads any unreferenced file is also invalid. Each run now
  sees a private folder holding only its referenced clips. These 17 runs were repeated under that rule.
  The other 271 runs read exactly their referenced clips and are kept. The first pass is kept as
  `pilot/out_holdout/gemini_arms_v1_shared_dir.jsonl`.
