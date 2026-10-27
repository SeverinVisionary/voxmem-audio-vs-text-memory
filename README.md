# Old audio versus text memory in frozen speech LLMs

A preregistered study on the VoxMem benchmark (8K histories). The question: when a speech assistant must recall
how something was said (who spoke, emotion, background sound) several sessions later, does keeping the old
user audio in memory beat spending the same KV positions on a text memory written from that audio?

Each split was run once under a preregistration written before it: `PREREGISTRATION.md` (Phase 1, test split,
300 families) and `PREREG_HOLDOUT.md` (Phase 1b, holdout split, 142 families). Both are shipped as frozen,
apart from one deviation note appended to `PREREG_HOLDOUT.md` §5. `PROVENANCE.md` gives the freeze commits
and file hashes. These come from the private repository the study was developed in, so the freeze is
self-attested; it lists the internal material the preregistrations mention that is not released.

## Results

All numbers are percentage points (pp) with 95% intervals, scored by the first judge unless stated.

**Phase 1 (test, MiniCPM-o 4.5): the H1 gate is not passed (decision NO-CUE).** Adding the whole old-audio
history on top of the best text memory shows no detectable benefit: F − T\* = −0.3 (−5.4 to +5.7),
p = 0.60. As preregistered, the matched-budget contrast is then reported as an estimate only: old audio at
matched KV positions is worth +0.7 (−3.2 to +5.6) over text, so a gain of 6 pp or more is excluded.
Every memory arm, text or audio, scores 27–31%; question only 8%; the full audio history (A) 28%.

**Phase 1b (holdout, MiniCPM-o 4.5).**
- B1: transcript-only memory is non-inferior to the full audio history at the preregistered 10 pp margin
  (34.5% vs 33.1%; M:asr − A = +1.4, −9.3 to +10.1): **TRANSCRIPT-SUFFICES** for this reader. The lower
  bound clears the margin by 0.7 pp.
- B2: no field group passes its one-sided test. Unplanned observation, in the opposite direction to the
  test: session-local speaker labels go with lower accuracy on speaker questions (−21.1, −34.6 to −11.8),
  but with the deterministic matcher this is −10.5 (−25.4 to +0.3), so it is scorer-sensitive.
- B3: F − T\* = 0.0 (−5.0 to +4.4); G − T\* = +2.8 (−1.2 to +7.1). With the second judge's verdicts
  substituted where it saw the answer, G − T\* = +4.2 (+0.4 to +8.3), so a small matched-budget gain for
  old audio on the holdout is not excluded.

**Strong-reader ceiling (48 holdout families; preregistered as exploratory).** Gemini 3.8 Flash, run through
the `agy` agent harness, gains from audio over the transcript: A − M:asr = +18.8 (6.6 to 31.7), against +2.1
for MiniCPM-o on the same families; difference between readers +16.7 (5.6 to 43.7). The text memory with
acoustic labels (T\*, 43.8%) matches full audio (A, 41.7%) for Gemini as well. Old audio on top of T\* adds
+2.1 at matched positions (G; −7.1 to +10.3) and +6.3 over budget (F; −6.4 to +25.7). At n = 48 these
intervals do not exclude a sizeable gain.

Reading: the benchmark's acoustic questions need acoustic information. Gemini extracts it from the audio and
MiniCPM-o mostly does not. A compact text memory with acoustic labels matches the full audio history for
both readers, at about 1/300 of the storage of the raw audio. It takes about 45% more KV positions, because
the labels are verbose. Whether old audio adds anything beyond such a memory is bounded for MiniCPM-o (under
6 pp at matched positions) but not settled for Gemini.

## Not claimed

- Retained-KV memory. Memory is re-prefilled for every question. The planned check that retained KV and
  re-prefill agree was not run, so all results describe re-prefill memory.
- Anything about readers other than these two, about Gemini outside the `agy` harness, or about benchmarks
  other than VoxMem 8K.
- A causal account of why MiniCPM-o makes little use of the audio.
- Public verification of the preregistration timing (see `PROVENANCE.md`).
- The frontier, the Phase 1 equivalence margin (5.6 pp) and the strata (`pilot/analyze_extra.py`), and the
  scoring-sensitivity analysis (`pilot/analyze_scoring.py`), are post hoc.

## Limitations

- **Scoring.** The first judge and the deterministic matcher disagree on 21–25% of answers outside the
  question-only arm: 15–29% per arm, a 14 pp spread across Gemini arms. Almost all of these are
  judge-correct, matcher-incorrect. The matcher is strict: it scores most memory arms at 4–13% on MiniCPM-o
  (A 4.7% on test).
  - Under the matcher, Phase 1 F − T\* is −3.3 (−7.9 to −0.2), favouring text.
  - The second judge saw 32–43% of answers: a seeded 10% plus every judge/matcher disagreement. Where it
    did, it agrees with the first on 98.6–100%.
  - Both preregistered decisions (NO-CUE, TRANSCRIPT-SUFFICES) and the sign of the Gemini A − M:asr gap
    hold under the second judge and under the matcher. All of this is in `pilot/out_holdout/SCORING_RESULTS.json`.
- **Departures from the preregistered judging.** The judge CLI has no temperature setting. The owner
  adjudication of disagreements was not carried out. The matcher is a fixed normalise-and-contain rule
  (`pilot/judge.py`), not VoxMem's per-item normalisation.
- **Tuning asymmetry.** On the dev split the text memory got about nine screened settings and the
  audio-selection policy two (`PREREGISTRATION.md` §7). This favours the text baseline.
- **Resampling units.** Families sharing a voice or a background clip were to be merged. That left a giant
  component (two components in all) on both splits, so the preregistered fallback, one source per family,
  was used: 48 units on test, 35 on holdout.
  - Voice purity was 0.905 on test and 0.891 on holdout, below the 0.9 target.
  - Clip-match accuracy was 0.72 (n = 36) on test and 0.58 (n = 19) on holdout.
- **Gemini harness.** The agent opens each clip with a file-read tool, so audio reaches the model as tool
  results rather than interleaved with the text. The model reasons over several steps, there is no
  temperature setting, and what audio representation reached it is not recorded. `agy` was version 1.4.7
  at release.
  - A first pass let runs open other families' clips. The 17 affected runs were repeated with per-run
    folders, as logged in `PREREG_HOLDOUT.md` §5.
  - Both runs that stayed invalid are transcript-only (M:asr): the agent kept calling shell commands.
    They are scored incorrect. Dropping those two families instead gives A − M:asr = +19.6, because both of
    their A answers were wrong.
  - One G answer is a safety-filter refusal and is scored as given (incorrect).
- **Smaller points.**
  - Assistant text turns are present in every history arm, A included.
  - At f = 0.5, 4 of 300 T\* items fall 43–534 positions short of budget because the text runs out. This
    affects a secondary contrast only.
  - The Phase 1 primary reader was changed from Ultravox-v0.6 to MiniCPM-o 4.5 before the freeze, because
    Ultravox was at floor on the dev split (`PREREGISTRATION.md` §10).

## Layout

- `pilot/`: the code.
  - Data extraction `data.py`; writer features `features.py`.
  - Memory records, budgets and rendering `memory.py`.
  - Readers `reader.py` and `gemini_reader.py`; runs `run_dev.py` and `diag.py`.
  - Source clustering `clusters.py`; judging `judge.py`.
  - Analyses `analyze_*.py`; unit tests `test_memory.py`.
- `pilot/out_dev/`: dev-split runs used for tuning (Ultravox and MiniCPM-o), with verdicts.
- `pilot/out_test/`: Phase 1 outputs, clusters, verdicts, `TEST_RESULTS.json` and `EXTRA_RESULTS.json`.
- `pilot/out_holdout/`: Phase 1b outputs, verdicts, `HOLDOUT_RESULTS.json` and `SCORING_RESULTS.json`.
  - Gemini outputs are in `gemini_arms.jsonl`; the first, shared-folder pass is in
    `gemini_arms_v1_shared_dir.jsonl`.
- `phase0/`: the frozen split, plus the benchmark metadata and prompts the code reads.
- `BYTE_ACCOUNTING.md`: ledger definitions. `PROVENANCE.md`: freeze record.

## Reproduce

`bash reproduce.sh` takes Python 3 with numpy; no GPU, no network. It reruns
- `analyze_test.py`, `analyze_extra.py`, `analyze_holdout.py` and `analyze_scoring.py`, on the shipped outputs
  and verdicts.

It then compares every number pinned in `expected.json` (all arm accuracies, contrasts, decisions, frontier,
strata and scoring sensitivity) and exits 3 on any difference.

`bash reproduce.sh --negative` flips one verdict in a copy and requires exit 3 there.

Not covered by `reproduce.sh`:
- the cluster statistics, which are read from `clusters.json`;
- the Gemini rerun counts, which are logged in `PREREG_HOLDOUT.md`.

Regenerating the outputs needs:
- a 24 GB GPU;
- MiniCPM-o 4.5 and Ultravox-v0.6 weights, transformers 4.51.3, torch 2.8.0;
- the benchmark audio, which `pilot/data.py` downloads (`python data.py OUT test`).

Then run `features.py`, `clusters.py`, and `run_dev.py --reader minicpm --stage arms --arms test
--settings pilot/frozen_minicpm.json`. Dataset and model revisions are not pinned, so regenerated outputs
may drift. Judging calls the two judge models named in `pilot/judge.py`. Raw judge transcripts are not
shipped; the per-answer verdicts are.

## Licence

Code (`*.py`, `reproduce.sh`): MIT (`LICENSE`). Data (`phase0/`, `pilot/out_*/`): CC BY-NC 4.0, derived from
VoxMem; see `DATA_LICENSE.md`, which also covers the documentation. Cite as in `CITATION.cff`.
