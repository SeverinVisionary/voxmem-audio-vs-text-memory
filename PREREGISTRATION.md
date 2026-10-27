# Q2 Phase 1 preregistration — v3 (2026-10-25)

Status: **frozen** by the commit recorded in §11 (v3: primary reader changed to MiniCPM-o 4.5 and H2
restated as an estimate before any test run; changes from v2 in §10). Nothing on the test split was run
before freezing. Plan: `PLAN.md` v1. Byte rules: `BYTE_ACCOUNTING.md`. Review of v1 and
dispositions: `reviews/prereg/`.

## 1. Question and hypotheses

For frozen speech-LLM readers and a causal text/vector memory, does native old audio held within a
matched **KV-resident** budget improve delayed speaker / paralinguistic / environmental recall over
spending the same KV positions on text? (Under a storage-form ledger the answer could differ; that
ledger is reported, not tested.)

- **H1 (gate: audio carries usable cue): ΔF = acc(F) − acc(T\*) > 0.** F = T\* plus the full user-audio
  history (over budget). Expected to pass; failure means the instrument cannot show an audio effect.
- **H2 (old audio vs text at equal positions): ΔG = acc(G) − acc(T\*) at cap f = 0.25**, tested one-sided
  (> 0) only if H1 passes. Because the dev freeze check failed twice (§11), H2 is also stated as an
  **estimate**: ΔG at f = 0.25 with its 95% interval is reported whatever H1 shows. Expected precision on
  test: interval half-width ≈ 4–6 pp (dev discordance 0.125, design effect 1.3–1.7).
- Secondary (exploratory): ΔG at f = 0.50 and 0.10; F − E; G − G_mismatched; per evidence type.

Estimand: family-weighted mean over the 300 test families (paralinguistic is 47% of the weight);
equal-weight-by-type reported as secondary. A null is reportable: the deliverable is the measured
positions → recall frontier either way.

## 2. Pre-pilot evidence (VoxMem 2609.32607, Fig. 4 and Table 2)

8K full-audio accuracy (%, **includes answer-refusal items**; recomputed on answerable dev items in the pilot):

| Reader | Speaker (n=147) | Paralinguistic (264) | Environmental (156) | Pooled |
|---|---|---|---|---|
| Ultravox-v0.6-Llama-3.1-8B | 42.2 | 15.2 | 15.4 | 22.3 |
| MiniCPM-o 4.5 | 38.1 | 21.2 | 17.9 | 24.7 |

The benchmark was filtered so a transcript does not answer acoustic items (Gemini-3.7-Flash: full audio
46.2% → transcript 4.4% on the 474 acoustic questions). T\* is stronger than a transcript, and G keeps
only part of the audio: expected ΔG ≈ (evidence coverage of G) × (gain when the evidence is present).
At f = 0.25 that may be a few pp, below the MDE; §7 makes the dev pilot measure it before freezing.

## 3. Data and splits

- VoxMem `AudioMemory/voxmembench` 8K (CC BY-NC 4.0). Unit = question family; one answerable question
  per family; no two acoustic families share an evidence session (Phase 0).
- Split frozen by `phase0/split.py` (seed `q2-acoustic-residual-2026-10-24`), `phase0/splits_8k.csv`,
  SHA-256 `08f48e3f8187a1e8f9cbf9c521fd3733e19863b9c68ca0a9911540d5a514a25f`:

  | Split | Speaker | Paralinguistic | Environmental | Use |
  |---|---|---|---|---|
  | dev | 8 | 15 | 9 | all tuning; includes the 18 GPU-smoke items |
  | test | 80 | 140 | 80 | primary analysis, run once |
  | holdout | 38 | 67 | 37 | Phase 2 only; never used to rescue Phase 1 power |

  Within type, allocation is proportional over memory operation. `subset96` = 96 test families (32 per
  type) for the secondary reader and 32K. 64 speech-semantics families (`semantic_sanity`) check that
  memory arms do not break ordinary recall.
- Real-audio transfer: Audio MultiChallenge, 42 audio-cue items; diagnostic only.

## 4. Readers and generation

- Primary: **MiniCPM-o 4.5** (`openbmb/MiniCPM-o-4_5`, audio only), through its own `chat()` with the
  flat single-message layout, SDPA attention, `logits_to_keep = 1` in generation. Position budgets are
  counted in its own positions (`memory.MiniCPMCounter`; equal to its real input length on every dev row).
  Ultravox-v0.6 was the v2 primary; it is at floor on dev (`pilot/DEV_RESULTS.md`) and is not run on test.
  No secondary-reader run in Phase 1 (`subset96` is dropped).
- transformers 4.51.3, torch 2.8.0, BF16, greedy, batch 1, `max_new_tokens` fixed on dev.
- System prompt: VoxMem `candidate_system_prompt.no_abstain.txt` plus one answer-format line, identical
  in every arm. Current question: identical audio, text-free, line `[Now <query_timestamp>]`.
- Session order and timestamps preserved; memory content placed where each session was.

## 5. Arms (byte rules in `BYTE_ACCOUNTING.md`)

**Causality:** a session's memory comes from that session's audio and earlier memory only; no
question, gold, evidence-role metadata, later turns or other items. Synthesis controls are never
exposed. **Writer leakage rules, asserted in code:** environment labels use AudioSet-label heads only
(no ESC-50-fine-tuned head, since ESC-50 is the background source); speaker ids in D′ come from online
clustering within the item over turns seen so far; transcripts in M are the writer's own ASR output,
never VoxMem's synthesis text.

| Arm | Content | Budget | Role |
|---|---|---|---|
| **T\*** | M + best extra text (dev-chosen queue) | B_f | comparator |
| **G** | M + old user-audio turns, online selection (dev-frozen: hash-priority eviction or streaming k-centre; capacity after each turn = ⌈f · audio seen so far⌉, discarded turns never return) | B_f | H2 |
| **G_mismatched** | G's slots and turn count, audio from another family of the same evidence type (donor's own G selection, highest-hash clips down to G's count; donors tried in hash order) | B_0.25 | cue check |
| **E** | M + most recent user-audio turns | B_f | secondary |
| **F** | T\*(B_0.25) + whole user-audio history | over budget | H1 gate |
| **A** | full audio history only (VoxMem setting) | — | anchors pipeline vs VoxMem |
| **N** | no memory (question only) | — | guess floor per type |
| C, D, D′ alone, oracle crop, shuffled tags, cue-neutralised | — | — | dev diagnostics |

f ∈ {0.25 primary, 0.50 secondary, 0.10 exploratory}. M = compact record frozen on dev: one of C
(Ultravox-v0.6 self-tags, an external tagger for the MiniCPM-o reader), D (SenseVoice + emotion2vec + AudioSet environment labels from AST on the first
10.24 s of each turn + session-local speaker labels from WavLM-SV) or D′ (D plus serialised relational features), with the writer's ASR transcript.

## 6. Outcome and scoring

- Binary correctness per family and arm, VoxMem answerable-judge prompt, one fixed judge
  (`gemini-3.8-flash-high` via `agy`, temperature 0), blinded to arm (shuffled, labels removed, tag
  vocabulary canonicalised). Deterministic normalisation (VoxMem `answer_normalization`) in parallel;
  every judge/deterministic disagreement plus a seeded 10% sample per arm goes to a second judge
  (`claude-opus-5-5`, low) and, on disagreement, to the owner, **also blinded to arm**. Judge
  disagreement rates are reported by arm; an arm-dependent rate above 5 pp is flagged as a threat.
- Reader errors: one rerun; persistent failure scored incorrect and logged. > 5% failures in any
  primary arm → run **INVALID**.

## 7. Analysis

- **Clustering (sources recovered before any test outcome is seen; `pilot/clusters.py`):** voice of
  each evidence-session user turn = one of 30 clusters from average-linkage cosine clustering of the
  WavLM-SV turn embeddings (reference-free; the VCTK 0.92 originals are an 11 GB archive); background
  clip of environmental families = the ESC-50 original (of the 45) whose normalised cross-correlation
  with the turn is ≥ 3× the next best. A family's sources = its sessions' majority voices and matched
  clips; families sharing any source are merged into connected components, the resampling units.
  Reported: voice purity (share of turns in their session's majority voice) and clip-match accuracy on
  families whose gold is a single clip category. If the largest component holds > 25% of the families,
  the units fall back to one source per family (majority voice; matched clip for environmental families).
  Clusters are computed in the test run before any judging; outcomes are not read until they exist.
- **Tests:** fixed sequence. H1 at one-sided α = 0.025; H2 tested only if H1 passes, at one-sided
  α = 0.025. p-values from a cluster-level sign-flip permutation test (10,000 flips); intervals from a
  type-stratified cluster bootstrap-t (10,000 resamples). Leave-one-cluster-out as sensitivity.
- **Decision (primary reader):**
  - H1 fails → **NO-CUE** (instrument null; audio not usable even in full).
  - H1 passes and ΔG interval U < 10 pp (unless H2 rejects with L > 0) → **TEXT-WINS** at matched
    positions (FUTILITY for old audio).
  - H1 passes, ΔG L > 0 and G − G_mismatched > 0 (one-sided α = 0.025) → **GO** = admit to Phase 2.
    GO does not claim a ≥ 10 pp effect.
  - Otherwise **INCONCLUSIVE**; no automatic expansion, no holdout rescue.
- **Power and freeze check:** 300 families, 80% power, one-sided α = 0.025: MDE 8.1 / 10.2 pp at
  discordance 0.25 / 0.40 unclustered; with test clusters of ≈ 7.3 families per voice and ≈ 1.8 per clip,
  ICC 0.05–0.10: **9.0–9.9 / 11.4–12.5 pp** (unequal cluster sizes push higher). Before freezing,
  compute expected ΔG = coverage(G, f) × (oracle-crop gain over T\*) on dev, using evidence metadata for
  this calculation only. If expected ΔG at f = 0.25 is below the MDE, promote f = 0.50 to primary before
  freezing, or restate H2 as estimating the frontier with a stated precision. Recorded in §11.
- Exploratory, labelled as such: per type, per memory operation, f = 0.10, F − E, secondary reader,
  32K subset, semantic sanity, AudioMC, retained-KV check (`BYTE_ACCOUNTING.md` §3). Oracle-crop
  success with F failure ⇒ interference rather than missing cue. Note for the write-up: T\* receives far
  more dev tuning (≈ 9 screened settings) than G (2); this favours the text baseline.

## 8. Run order and dev-pilot measurements

1. Dev pilot on the 32 dev families, measuring:
   G and E evidence coverage per cap and the oracle-crop gain (→ expected ΔG); discordance F vs T\* and
   G vs T\* (replaces assumed 0.25 / 0.40); voice-assignment purity and clip-match rate; judge vs
   deterministic vs owner agreement by arm; zero-audio-turn rate of G, P_A distribution, fill ratios;
   refusal and truncation rates by arm; arm A accuracy on answerable items vs §2; G vs G_mismatched;
   checkpoint provenance (AudioSet heads, Llama weights); MiniCPM-o audio positions per second;
   batched tag-writing throughput; retained-KV vs re-prefill agreement.
2. Fill §9, run the freeze check in §7, freeze (commit SHA in §11), post the SHA on #1082.
3. Source clustering on all 474 families (audio features only; no outcomes).
4. Test run, primary reader, once: N, A, F, T\*/G/E at f = 0.25, 0.50, 0.10, G_mismatched@0.25
   (`pilot/q2_minicpm_test.ipynb`; the 18 dev rows that ran out of memory are rerun first as a gate).
5. Score (`pilot/judge.py`), audit, analyse with `pilot/analyze_test.py` (committed before step 4).

## 9. Set on dev, frozen before test (blank until the pilot)

Selection score: mean teacher-forced log-probability of the gold answer per family (accuracy breaks
ties, after judging). The retained-KV comparison (`BYTE_ACCOUNTING.md` §3) runs in a separate dev
notebook before freezing. Llama provenance: without an HF token for Meta's repo the mirror is used and
the provenance check is recorded as not run. Screen one factor at a time in the order below, holding the others at their default.

| Setting | Candidates | Frozen 2026-10-25 (MiniCPM-o; `pilot/frozen_minicpm.json`) |
|---|---|---|
| Writer for M | C, D, D′ | D (−15.12 vs C −15.95, D′ −15.13 mean gold log-prob) |
| Tag format | JSON one-line, key=value, prose | key=value (−15.12 vs −15.21, −15.22) |
| T\* extra-text queue | longer tags, relational features, prosody numbers (per-turn LLM summaries dropped: one more generation per turn) | prosody numbers (−15.01 vs −15.12, −15.10) |
| G selection | hash-priority eviction, streaming k-centre on audio + speaker embeddings | hash-priority eviction (−14.92 vs −15.09) |
| max_new_tokens | 32, 48, 64: generate at 64, greedy prefixes give the 32/48 truncation rates exactly; smallest with < 2% truncated | 48 (> 48 new tokens: 0% in every arm; > 32: up to 6.2%) |
| Answer-format line | 2 phrasings on T\*(0.25); fewer judge disagreements, applied to every test arm | line 0 (judge 1 vs judge 2 agreement 100% for both lines, a tie; broken by accuracy, 43.8% vs 31.2%) |

Ultravox-selected values (v2, superseded): writer C, key=value, longer tags, hash-priority, 64 tokens.

## 10. Deviations

Every change after freezing is logged with date, reason, and whether test outcomes had been seen.
Changes after seeing test outcomes make the affected analysis exploratory.

Before freezing v3 (2026-10-25; no test outcome seen, nothing run on test):
- Primary reader Ultravox → MiniCPM-o 4.5 (flat layout); Ultravox at floor on dev, MiniCPM-o lifts it
  (instrument check in `pilot/DEV_RESULTS.md`). Budgets recounted in MiniCPM-o positions; §9 re-screened
  with it. Secondary reader and `subset96` dropped.
- H2 also stated as an estimate with its precision (§1), the §7 remedy for a failed freeze check.
- Clustering: reference-free voice clusters instead of VCTK reference centroids, an explicit
  giant-component fallback, and the decision order TEXT-WINS vs GO made explicit (§7).
- `max_new_tokens` 48; the answer-line rule's tie broken by accuracy (§9).

## 11. Freeze record

Expected ΔG check, Ultravox (dev pilot, 2026-10-25, `pilot/DEV_RESULTS.md`): **failed.** Coverage of G@0.25 = 90.6%,
oracle-crop gain over T\* = −3.1 pp, expected ΔG ≤ 0 at every cap (MDE 9.0–9.9 pp); ΔF = 0.

Expected ΔG check, MiniCPM-o (dev, 2026-10-25, `pilot/out_dev/DEV_RESULTS_minicpm.json`): **failed.** T\*@0.25
43.8%, G@0.25 43.8% (ΔG = 0, discordance 12.5%); coverage of G@0.25 = 90.6%, oracle − T\* = −3.1 pp, expected
ΔG = −2.8 pp at f = 0.25 (−3.1 at 0.50). ΔF = −12.5 pp on the 24 families without out-of-memory failures
(discordance 20.8%). G@0.50 − T@0.50 = +12.5 pp (n = 32). Remedy taken: H2 restated as an estimate (§1); f = 0.25
stays primary (promoting 0.50 on the strength of one dev contrast would be selection on dev noise).
Expected outcome from dev: NO-CUE if F stays below T\*, with ΔG bounded.

Primary cap: f = 0.25 · Frozen SHA: (the commit that adds this line; recorded on #1082) · Date: 2026-10-25
