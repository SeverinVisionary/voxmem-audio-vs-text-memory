# Provenance of the preregistrations

The study was developed in a private repository. This public repository is a single snapshot made after both
runs, so its own history cannot show that anything was fixed in advance. What follows is the record from
the private repository; it is self-attested, and nothing here is independently timestamped.

| Event | Private commit | Time (US Pacific) |
|---|---|---|
| Phase 1 preregistration v3 frozen | `e96d7e1dd6a3057df7c02110a7e7b86cc543b86e` | 2026-10-25 16:49 |
| Phase 1 test outputs committed, before judging | `5d60788678b9cdd03a21a1c2717ae86f2acd63ac` | 2026-10-25 22:55 |
| Phase 1b preregistration frozen | `e10bc521f87811f3425113fc8f409197aa0d6b5f` | 2026-10-26 08:04 |
| Phase 1b MiniCPM-o outputs committed, before judging | `f9b3c7e940c9ea3c19a10cfaf6d5f57b8879fa0e` | 2026-10-26 10:55 |

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

`PREREG_HOLDOUT.md` differs from its frozen form by one appended bullet under §5 (the Gemini deviation,
starting "2026-10-26, before any Gemini answer was judged"), added after the freeze as that section requires.
Deleting that bullet and the blank line before it restores the frozen bytes.

Both preregistrations refer to material that is not released: planning and review notes (PLAN.md,
reviews/), internal write-ups (pilot/DEV_RESULTS.md, pilot/TEST_RESULTS.md), the split script
(phase0/split.py), the Colab notebooks and their builders, and a private issue tracker (#1082). The
dev-split outputs those write-ups summarise are in `pilot/out_dev/`.
