# Provenance of the preregistrations

The study was developed in a private repository. This public repository was first published after the first two
studies and updated after the third, fourth and fifth, so its own history cannot show that anything was fixed in advance. What follows is the record from
the private repository; it is self-attested, and nothing here is independently timestamped.

| Event | Private commit | Time (US Pacific) |
|---|---|---|
| Phase 1 preregistration v3 frozen | `e96d7e1dd6a3057df7c02110a7e7b86cc543b86e` | 2026-10-25 16:49 |
| Phase 1 test outputs committed, before judging | `5d60788678b9cdd03a21a1c2717ae86f2acd63ac` | 2026-10-25 22:55 |
| Phase 1b preregistration frozen | `e10bc521f87811f3425113fc8f409197aa0d6b5f` | 2026-10-26 08:04 |
| Phase 1b MiniCPM-o outputs committed, before judging | `f9b3c7e940c9ea3c19a10cfaf6d5f57b8879fa0e` | 2026-10-26 10:55 |
| Qwen3-Omni preregistration frozen | `b900cf41227a520fdaf4f9f77d75f8ef29dc0c8b` | 2026-10-28 18:28 |
| Qwen3-Omni outputs committed, before judging | `dc3a14f0bc742db8918cd15434529f91b45c27fb` | 2026-10-28 21:11 |
| Positive-control preregistration frozen | `ea88e3591fb89ee296574daef6100b7430e8a269` | 2026-11-02 08:44 |
| Positive-control outputs committed, before judging | `80ee194317917d24285cb37678b0207dffa46ba8` | 2026-11-02 09:00 |
| Second positive-control preregistration frozen | `5e7d36b03b7d04ea711d7098572186ab92f88028` | 2026-11-02 15:31 |
| Second positive control, MiniCPM-o outputs committed, before judging | `385beb3c98d7e1c1851a3b46ffb8e568a8775cef` | 2026-11-02 16:00 |
| Second positive control, Gemini outputs committed, before judging them | `035c0b63096537f2e7d795b878c8b771226026a8` | 2026-11-02 19:00 |

SHA-256 of the files as frozen; the shipped copies are byte-identical except where noted.

| File | Frozen at | SHA-256 |
|---|---|---|
| `PREREGISTRATION.md` | e96d7e1 | `be18ac392655c3d3c9301ccc019f30f54ac73da8cc0718f502dc761a3a1df791` |
| `BYTE_ACCOUNTING.md` | e96d7e1 | `8c2893a37fd5db37ac647060082133cd19ff4ae4f6bd9316c1aaed87a7bf36ce` |
| `pilot/analyze_test.py` | e96d7e1 | `646ed37dddd1f2317354dc87c02a55fb8d90fd59467a46c34f1bdbfd4de02989` |
| `pilot/frozen_minicpm.json` | e96d7e1 | `4a385c8d0b165c208c61fc8d88d85e1b6b0486d1af938759d797ff440b7f44df` |
| `phase0/splits_8k.csv` (CRLF line ends) | before v1 | `08f48e3f8187a1e8f9cbf9c521fd3733e19863b9c68ca0a9911540d5a514a25f` |
| `pilot/analyze_holdout.py` | e10bc52 | `0a19269b8ecee8c4b07c0ae8066967ceb520cb14744c9d4dba3eb0eda1c01c20` |
| `pilot/memory.py` as used for Phase 1 (not shipped; later versions add the Phase 1b writers) | e96d7e1 | `ec814669409f3ac034c748cc41ba7e81f9265e7578608deffd46372b3d9c6203` |
| `pilot/memory.py` | e10bc52 | `82c2b2642269c5a5ac19ff2f1f567cdd8d9a8dd2ce376633555f13a3082558ff` |
| `PREREG_HOLDOUT.md` | e10bc52 | `5ad2e8358e61283f26a181aa63ce2a8399497f185606db4ab642c6003290c3fc` (see below) |
| `PREREG_OMNI.md` | b900cf4 | `8ceaa8798d30414b015ae00ebdcf579708d178c00fb02c4676666e938e04a3c3` (see below) |
| `pilot/omni_reader.py` | b900cf4 | `45c5fa65647afc090c15da44102db95800d35a674dc8aeb1e3c9363bf9eadf3f` |
| `pilot/analyze_gemini_clean.py` | b900cf4 | `e3106a39882e3a5d6874ebe7a9ba7826b7838786a94b0bc1646f99b1b099365c` |
| `PREREG_POSCTRL.md` | ea88e35 | `4fcbff653fbb83dc9fbc72ef39360d02395060ab51e54122ee6c71600054e748` (see below) |
| `pilot/make_posctrl_plans.py` | ea88e35 | `84deb68910a1b808a0abc7c5d21ce998b3285db863877f9fc06971fc382bfb0c` |
| `pilot/analyze_posctrl.py` | ea88e35 | `145a21749616086e94c1d5810964d95854356350a0edceb6c513cd4676face2c` |
| `PREREG_POSCTRL2.md` | 5e7d36b | `4391a7e562312928c1ea6e3bd40c949ee09d8b819798d3fe90f5f8b8af7f5d88` (see below) |
| `pilot/make_posctrl2_plans.py` | 5e7d36b | `fc4560d8f237f44a0a6c8a093ad9c52282d8291fcaa0a04637bcaace062ccd24` |
| `pilot/minicpm_plans_reader.py` | 5e7d36b | `89901b9661ac90cbd489a7091ae869c94e72b1542b1c811ff1d104f9b5af9a7c` |
| `pilot/analyze_posctrl2.py` | 5e7d36b | `d620b1008c4038715d408084a9d2dd6c00e8cd94ea33fa51d01a4a5040b9bf2e` |

`PREREG_HOLDOUT.md` differs from its frozen form by one appended bullet under §5 (the Gemini deviation,
starting "2026-10-26, before any Gemini answer was judged"), added after the freeze as that section requires.
Deleting that bullet and the blank line before it restores the frozen bytes. `PREREG_OMNI.md` likewise
differs by one appended §5 bullet (starting "2026-10-28, before any answer existed"), and deleting it with the
blank line before it restores the frozen bytes. `PREREG_POSCTRL.md` differs in §6 only: the frozen text there
is the line "(none yet)", replaced after judging by one bullet (starting "2026-11-02, after the answers were
judged"); putting that line back restores the frozen bytes.
`PREREG_POSCTRL2.md` differs in two ways, so its hash cannot be checked from the shipped copy. Its §6, frozen
as "(none yet)", now holds the two deviation notes written during the run. In §2, the phrase naming the
private repository's checkout among the folders the sealed Gemini run could not read is replaced by "the
private source checkout".

The preregistrations refer to material that is not released: planning and review notes (PLAN.md,
reviews/), internal write-ups (pilot/DEV_RESULTS.md, pilot/TEST_RESULTS.md), the split script
(phase0/split.py), the Colab notebooks and their builders, a private issue tracker (#1082), the abandoned sealed Gemini rerun's
preregistration (PREREG_GEMINI_CLEAN.md) and the sealed plan archive the Qwen3-Omni run read
(plans_clean.zip; it is regenerable with `run_dev.py`). The positive control's plans are derived from that
archive by `pilot/make_posctrl_plans.py`, and the second positive control's from both by
`pilot/make_posctrl2_plans.py`; the Gemini run's sandbox launcher is not released. The
dev-split outputs those write-ups summarise are in `pilot/out_dev/`.
