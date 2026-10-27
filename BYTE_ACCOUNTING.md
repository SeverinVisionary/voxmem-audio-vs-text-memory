# Q2 byte accounting (v2, 2026-10-24)

Defines the persistent-memory ledger that every arm is matched on. Referenced by `PREREGISTRATION.md` §5.

## 1. What is counted

**Primary ledger — KV-resident bytes.** Everything the reader can attend to at question time that was
carried over from earlier sessions, counted as the KV it occupies in that reader:

    bytes = Σ_segments positions(segment) × β_reader  +  overhead

| Reader | β (BF16 K+V per position) | Audio positions / s |
|---|---|---|
| Ultravox-v0.6-Llama-3.1-8B | 32 layers × 8 KV heads × 128 × 2 (K,V) × 2 B = **131,072 B** | 6.25 |
| MiniCPM-o 4.5 (Qwen3-8B LLM) | 36 × 8 × 128 × 2 × 2 = **147,456 B** | measured on dev (≈ 10 audio tokens/s plus unit-boundary positions) |

- **Segments:** memory text (transcript, tags, relational features, serialised exactly as the reader
  sees them, tokenised by the reader's tokenizer), old-audio positions, recent-audio positions,
  session headers/timestamps, and any separator or template tokens the memory adds. Wrapper tokens
  (audio placeholders, speaker/turn wrappers, timestamp lines) are charged to the arm that adds them;
  the ledger code asserts that the counted positions equal the reader's actual input length minus the
  shared prefix.
- **Overhead:** quantisation scales and zero-points (if any arm quantises), allocated-but-unused page
  slots when a paged cache is used (count allocated blocks, not filled positions), and index/position
  tables kept with the memory. With the Phase 1 HF implementation (contiguous BF16, no quantisation)
  overhead is 0 and is reported as such.
- **Excluded (identical in every arm, reported once):** system prompt, the current question's audio and
  text, answer-format instruction, model weights.

Because β is constant within a reader, matching bytes within a reader is the same as matching
**positions**. All budgets below are stated in positions and converted to bytes for reporting.

**Secondary ledger — storage-form bytes (reported, not used for decisions).** The same memory kept in
its cheapest storage form: text as UTF-8; audio as 16 kHz PCM16 (32,000 B/s); embeddings at their
stored dtype; plus the read-time re-encode cost (prefill positions and wall seconds) it implies. This
is the ledger a "stored-PCM/embedding replay" system would use; it is what makes text look ≈ 10⁵×
cheaper per byte than KV, so it is shown beside the primary ledger and never mixed with it.

## 2. Budgets

For item *i* and reader *r*: P_A(i) = audio positions of the full **user**-audio history (assistant
turns are text and are part of every arm's transcript); P_M(i) = positions of the compact record **M**
frozen on dev. The same P_A definition is used for F.

    B_f(i) = P_M(i) + ⌈f · P_A(i)⌉,   f ∈ {0.25 primary, 0.50 secondary, 0.10 exploratory}

Budgets are per reader (MiniCPM-o has more audio positions per second than Ultravox); results are not
compared across readers on a byte axis.

- **T\***: M plus the best extra text/vector content chosen on dev, truncated or padded with the next
  dev-chosen text item until it uses B_f(i) ± 1 position.
- **E / G**: M plus whole user turns of audio (most recent first for E; for G the dev-frozen
  **online** rule: reservoir-uniform or streaming k-centre, decided at write time without knowing the
  final history length), added while the total stays ≤ B_f(i). Positions left over (less than one turn)
  are filled with the same extra text T\* would use next, so every budgeted arm totals B_f(i) ± 1.
  The audio/text split of each arm is logged per item.
- **F**: T\* at B_0.25 plus the whole audio history. Over budget by design; diagnostic of whether audio
  carries usable cue, not a ceiling.
- Items where P_A(i) < 4 audio turns are kept but flagged; per-item fill ratios and the number of
  items where G holds **zero** audio turns at each cap are reported (on those items G equals T\* up to
  padding, which dilutes ΔG).
- All fill text for T\*, G and E comes from the same dev-frozen queue, in bit-reversal turn order. This
  padding is decided at query time (it uses the history length), identically for every budgeted arm.

## 3. KV retention versus re-prefill

The Phase 1 implementation **re-prefills** the selected memory (text and audio, chronological order) for
every question with a frozen reader, and charges it at the positions it would occupy if retained.
This is exact for the primary ledger only if attending to retained KV and re-prefilling give the same
answers. The dev pilot checks it: for 32 dev items × arm G, compute (a) old-audio KV cached at write time
in its original session context, spliced in with positions re-indexed, and (b) re-prefill of the same
clips. Re-indexing applies the RoPE rotation for the new positions to the cached keys explicitly.
Report answer agreement, correctness agreement, and the first-answer-token KL divergence. If answer
agreement is < 90% or mean KL > 0.05,
Phase 1 results are labelled "re-prefill memory", not "retained-KV memory", and Phase 2 is not admitted
until a retained-KV implementation passes.

## 4. Reported alongside (never matched)

Peak read-time GPU memory, read-time prefill positions, time to first token, end-to-end seconds per
question, and write-time cost per session (tag-generation seconds, encoder seconds).
