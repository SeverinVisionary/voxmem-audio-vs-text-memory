# Q2 — open audio reader (preregistration, frozen before any run)

Written 2026-10-28. The clean Gemini rerun (`PREREG_GEMINI_CLEAN.md`) was abandoned at 324 of 710 runs on
quota, unscored, and the cross-reader claim was dropped. The readers reachable through the same tooling (Claude
Opus 5.5, GPT-6 Astra) cannot hear audio. This study asks the same question of an open audio reader that we run
ourselves, so the answers are out of its reach by construction. None of its answers exist yet.

## 1. Question

Does a second, larger audio reader gain from old native audio over a transcript at the matched budget, as
Gemini appeared to in Phase 1b and MiniCPM-o did not?

## 2. Design

- Reader: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (Hugging Face, `transformers==4.57.1`), thinker only, bf16, sdpa,
  greedy, `max_new` = 48 from the Phase 1 frozen settings. Code: `pilot/omni_reader.py`, notebook
  `pilot/q2_omni.ipynb` (one Colab run on an 80 GB GPU, run-once guarded).
- Inputs: the 710 sealed plans of the clean rerun, unchanged (`plans_clean.zip`, sha256 a9c0b27b…): all 142
  holdout families × N, A, M:asr, T\*@0.25, G@0.25, rendered with MiniCPM-o budgets as for Gemini. The system
  text is the system turn; the memory text and clips alternate in one user turn where the plan has `<|audio|>`.
- Sealing: the plans carry no gold; the notebook never copies `gold.jsonl`. Gold is joined locally after the
  rows are downloaded (`omni_reader.py --join`).
- Failures: a row that errors (e.g. out of memory) is retried by the rerun stage; one still failing is scored
  wrong, and the count is reported.
- Scoring: judge 1 `claude-opus-5-5`, judge 2 `claude-sonnet-5-5` on a seeded 10% plus every judge-1/matcher
  disagreement (`judge.py`, `Q2_J1`/`Q2_J2`), deterministic matcher alongside. Gemini, judge 2 for the earlier
  readers, is out of quota.

## 3. Analysis (`pilot/analyze_gemini_clean.py`, run on this reader's rows)

Type-stratified cluster bootstrap-t with the holdout dominant-source units, one-sided sign-flip p, as before.

- **C1 (primary).** A − M:asr > 0 on the 142 families, one-sided α = 0.025: the 95% interval's lower bound
  above 0 → AUDIO-BEATS-TRANSCRIPT, otherwise NOT-SHOWN.
- C2 (estimate). T\*@0.25 − A; called within 10 points only if the whole interval lies in (−10, +10).
- C3 (estimate). G@0.25 − T\*@0.25. A − N is reported as a check that the reader hears the clips at all.
- C4 (estimate). (A − M:asr) Qwen3-Omni minus MiniCPM-o, under each reader's first judge and under the
  matcher for both.
- The script's "never_seen" block (families outside the Phase 1b Gemini subset) is reported but carries no
  special meaning for this reader.

## 4. What follows from it

- AUDIO-BEATS-TRANSCRIPT: the paper may say that audio memory helps a stronger open reader though not
  MiniCPM-o, as confirmatory for this reader and harness; C2 decides whether "labelled text matches audio" may
  be said for it.
- NOT-SHOWN: neither open reader gains from audio over a transcript at this budget; the paper says so, with
  the Gemini Phase 1b gain as an unreplicated exploratory result.

## 5. Deviations

Logged here with date and whether any answer had been seen.

- 2026-10-28, before any answer existed: the first launch stopped at the Google Drive mount (its sign-in
  popup was out of reach), and the runtime was released before any model ran. The notebook now takes the
  sealed plans from a private release asset (`q2-omni-plans-v1/plans_clean.zip`, same sha256) and commits
  the rows to branch `q2-omni-results` (`pilot/out_omni/`) after each stage, so no Drive is used. The GPU
  may be any card with more than 75 GB (Colab assigned a G4, RTX PRO 6000 96 GB, when no H100 was free).
  Reader, inputs, decoding and analysis are unchanged.
