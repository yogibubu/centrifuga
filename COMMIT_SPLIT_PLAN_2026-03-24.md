# Commit Split Plan (2026-03-24)

Use a narrow commit sequence that matches the current active project set.

## 1. `feat(gui): integrate shared vibro-rotational workspace`

Include:

- `ceditt_gui.py`
- GUI-facing documentation updates
- integrated alpha/quartic/sextic reporting

## 2. `feat(alpha): add quasiparticle alpha workflow`

Include:

- `vibrot_alpha.py`
- `scripts/alpha_channel_audit.py`
- alpha-related tests
- any alpha-support backend changes

## 3. `feat(vpt4): continue quartic-channel extension`

Include:

- `distortion_workflow.py`
- `rovib_distortion.py`
- `channel_contributions.py`
- `compare_gaussian_sextic.py`
- `harmonic_convention.py`
- `symmetry_metadata.py`
- VPT4/quartic tests

This commit line includes the still-open linear-molecule work.

## 4. `refactor(data): canonicalize gaussian benchmark inputs`

Include:

- `gaussian_vpt_parser.py`
- benchmark-path test updates
- `gaussian/`
- `archive/benchmarks_root_legacy/`

## 5. `docs: declare active projects and active manuscripts`

Include:

- `README.md`
- `DOCUMENTAZIONE.md`
- `ACTIVE_PROJECTS_2026-03-24.md`
- `REPO_STATUS_TRIAGE_2026-03-24.md`
- `manuscripts/active/README.md`
- `manuscripts/active/paper2.tex`
- `manuscripts/active/paper2.bib`

## 6. `chore(repo): remove retired manuscript and cache material`

Include:

- deletion of retired root manuscript files
- deletion of generated caches and transient packaging directories
- archival moves under `docs/` and `archive/`

## Not part of the active set

Do not spend additional cleanup effort on material unless it contributes to:

- `CeDiTT`
- `alpha_resonances`
- `VPT4 quartics`
