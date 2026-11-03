# Q2 — positive control on two more readers (preregistration, FROZEN 2026-11-02)

Drafted and frozen 2026-11-02, after the Qwen3-Omni positive control (`PREREG_POSCTRL.md`, P1 NOT-SHOWN), before
any answer for the arms below existed. Plans script, reader script, notebook builder and analysis script are
committed with this file.

## 1. Question

The Qwen3-Omni positive control did not show that the reader uses old clips even when given only the sessions
that hold the answer. One reader cannot carry a claim about current audio readers. This study asks the same
question of the other two readers in the paper:

1. **MiniCPM-o 4.5** (open, 8B, the primary reader of v1–v3). With only the evidence sessions in context, does
   audio beat their transcript? Inside the full transcript, does adding the evidence clips beat the transcript?
2. **Gemini 3.8 Flash** (closed, through `agy`). With only the evidence sessions in context, does audio beat
   their transcript?

## 2. Design

- Families: the same 142 holdout families; units: the holdout dominant-source clusters (`out_holdout/clusters.json`).
- Plans: no new rendering. `pilot/make_posctrl2_plans.py` takes N and M:asr from the sealed clean-rerun plans
  (`plans_clean.zip`, sha256 a9c0b27b…) and Ev-A, Ev-M, O from the positive-control plans
  (`q2-posctrl-plans-v1/plans_posctrl.zip`, sha256 ea255dc2…), byte-identical to what Qwen3-Omni read. Arm
  definitions as `PREREG_POSCTRL.md` §2. Families in hash order, a family's arms adjacent; no gold. The
  MiniCPM-o set is release asset `q2-posctrl2-plans-v1/plans_posctrl2_minicpm.zip` (sha256 591e93b7…).
- **MiniCPM-o**: arms N, M:asr, Ev-A, Ev-M, O (710 rows), all run fresh, so every contrast compares rows from
  the same run (no reuse, no replay rule). `pilot/minicpm_plans_reader.py`: the v3 reader class
  (`reader.MiniCPM`, flat layout, `chat()`, greedy, `max_new` from `frozen_minicpm.json`) on the exported WAV
  clips. Colab L4, the v1–v3 MiniCPM-o software pins. One rerun stage for failed rows; a row still failing is
  scored wrong.
- **Gemini 3.8 Flash** (`gemini-3.8-flash-high`): arms Ev-A, Ev-M (284 rows). `pilot/gemini_reader.py` in
  sealed mode exactly as `PREREG_GEMINI_CLEAN.md` §2 (per-run clip folder, `sandbox-exec` read denial of the
  plans, the gold vault, the private source checkout, `~/Downloads`, the session scratchpad and `~/.claude/projects`;
  canary must be blocked; trace rule; invalid twice → error, scored wrong). Run with `pilot/run_gemini_posctrl2.sh`
  in 4 shards. No O arm: it would need a sealed Gemini M:asr on all 142 families, which does not exist.
  A quota stop is resumed after the reset. If the 284 rows are not complete by 2026-11-09, P1-Gemini is reported
  as an estimate on the families finished, not as a test.
- Judging: `judge.py` on each reader's new rows after joining the gold (`omni_reader.py --join`), with each
  reader's v3 judges: MiniCPM-o judge 1 `gemini-3.8-flash-high`, judge 2 `claude-opus-5-5`; Gemini judge 1
  `claude-opus-5-5`, judge 2 `gemini-3.8-flash-high` (seeded 10% plus every judge-1/matcher disagreement).
  Deterministic matcher alongside.

## 3. Analysis

`pilot/analyze_posctrl2.py` on the gold-joined arms files and their verdicts. Type-stratified cluster
bootstrap-t, one-sided sign-flip p, seeds 4001 (MiniCPM-o) and 4101 (Gemini); unstratified bounds reported next
to each test.

- **P1-MiniCPM, P1-Gemini (primary, one per reader). Ev-A − Ev-M > 0**, one-sided α = 0.025 each. Lower bound
  above 0 → HEARS-OLD-CLIPS; otherwise NOT-SHOWN. No correction across readers: it would make a pass harder,
  and the paper's claim at stake is that no reader passes.
- **P2-MiniCPM (fixed sequence after P1-MiniCPM). O − M:asr > 0**, as `PREREG_POSCTRL.md` §3; reported as an
  estimate if P1-MiniCPM fails.
- Estimates: P1 by evidence type; MiniCPM-o Ev-A − N and Ev-M − N; agreement of this run's MiniCPM-o N and
  M:asr answers with v3's (same plans; differs only by the WAV round trip of the question clip).
- Power: as `PREREG_POSCTRL.md` §3, about 9–10 pp at 80% on 142 families.

## 4. What follows from it

| Readers passing P1 (of Qwen3-Omni, MiniCPM-o, Gemini) | Reading for the paper |
|---|---|
| none | Across three readers, two open and one closed, none is shown to use old clips even when isolated. The audio-vs-transcript null describes current readers, not the recordings. |
| Gemini only | Old clips are usable by a closed reader but not by the open ones. The open-reader null is an instrument null; Gemini's earlier exploratory audio gain becomes the case to test with a sealed full-history run. |
| an open reader | That reader can hear old clips; its v3 A ≈ M:asr then reflects use of history, and P2 says whether the right clips help amid it. |

## 5. Compute

- MiniCPM-o: one Colab L4 run, run-once guarded. v3's 1,420 holdout rows (10 arms, many longer than these)
  took about 2.5 L4-h with gold log-probabilities; these 710 rows generate only. Estimate under 1 L4-h,
  **about 2 CU**.
- Gemini: 284 `agy` runs plus up to one retry each on the operator's quota; ~600 judge calls in all.

## 6. Deviations

1. **Judge harness (2026-11-02, before any verdict of this study was read).** On one MiniCPM-o judge-1 batch
   (`j1_002`), `agy` in text mode printed its complete JSON answer and then never exited, so `judge.py` timed out
   six times. `judge.py` now parses what the process printed before the timeout; the existing check that every
   item id comes back is unchanged, so a cut-off answer is still rejected. Verdict rules, prompts and models are
   unchanged.
2. **Gemini invalid rows (no change to the plan, reported for reading).** The trace rule marked 4 Ev-M rows
   invalid twice (q_00227, q_00405, q_00427, q_00756: tool calls or file reads outside the clip folder). As §2
   says, they are scored wrong. A further 8 Ev-M rows passed the trace rule but ended with an empty reply after
   the model ran commands and web searches; `judge.py` crashed on the empty answer and now passes it to the
   judges as an empty response, which is wrong. All 12 are Ev-M, so they lower Ev-M and make P1-Gemini easier
   to pass; a sensitivity dropping those 12 families is reported next to the test.
