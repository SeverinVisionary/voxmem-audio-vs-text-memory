# Data licence

All files under `phase0/` and `pilot/out_*/` are derived from the VoxMem benchmark
(Hugging Face dataset `AudioMemory/voxmembench`, https://huggingface.co/datasets/AudioMemory/voxmembench,
CC BY-NC 4.0): benchmark item metadata, gold answers, the benchmark's prompts, and model outputs and
verdicts on its items. They are released under **CC BY-NC 4.0**
(https://creativecommons.org/licenses/by-nc/4.0/). Credit the VoxMem authors, the sources VoxMem credits
(the VCTK corpus for speech and ESC-50 for background sounds), and this repository.

Changes made: items were selected and split (`phase0/splits_8k.csv`); model answers, judge verdicts and derived statistics were added. Local file paths quoted inside
Gemini's transcripts (`raw` fields) were replaced with `<scratch>` or `<tmp>`, keeping the clip file names. No benchmark audio is
redistributed: `pilot/data.py` downloads it from the source above. ESC-50 per-clip licences are listed in
`phase0/out/vox_metadata__environmental_sources.json`; `pilot/clusters.py` downloads the ESC-50 originals
at run time and does not redistribute them.

The README, preregistrations, `PROVENANCE.md`, `BYTE_ACCOUNTING.md`, `expected.json` and
`pilot/frozen_minicpm.json` are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
