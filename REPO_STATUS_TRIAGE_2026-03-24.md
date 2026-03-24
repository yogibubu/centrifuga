# Repo Status Triage (2026-03-24)

This note summarizes the remaining dirty areas in the repository after the
vibro-rotational GUI integration, release refresh, and benchmark-data cleanup.

## Current safe state

- Active project set reduced to:
  - `CeDiTT`
  - `alpha_resonances` (external to this repo)
  - `VPT4 quartics`, including unfinished linear-molecule work
- Canonical Gaussian benchmark data is now under `gaussian/`.
- Root-level `.log` and `.fchk` files have been removed from the repository root.
- Legacy root benchmark mirrors were archived under `archive/benchmarks_root_legacy/`.
- GUI documentation was moved under `docs/gui/`.
- Legacy app bundles were moved under `archive/apps/`.
- Temporary merge TeX fragments were moved under `archive/tmp_merge/`.

## Remaining dirty areas

### 1. Code and tests

These are active development files and should be reviewed as normal code
changes, not treated as cleanup noise:

- `ceditt_gui.py`
- `vibrot_alpha.py`
- `gaussian_vpt_parser.py`
- `distortion_workflow.py`
- `channel_contributions.py`
- `compare_gaussian_sextic.py`
- `derive_watson_quartic_vanvleck.py`
- `harmonic_convention.py`
- `rovib_distortion.py`
- `symmetry_metadata.py`
- selected tests under `test_*.py`
- new diagnostics under `scripts/`

Recommended action:
- review and commit these as the vibro-rotational integration work

### 2. Tracked build artifacts

The following are tracked in git history and therefore still appear as changes
even though they are build/cache outputs:

- `.pyinstaller/...`
- `build/...`
- `dist/CeDiTT1.0.app/...`
- `dist/CeDiTT1.0/...`
- previously tracked `__pycache__/...`

Recommended action:
- decide whether release binaries should remain versioned
- if not, remove them from git in a dedicated cleanup commit
- if yes, keep only the current release outputs and stop tracking transient
  caches such as `.pyinstaller` and `build`

### 3. Manuscript and notes material

A large block of TeX/Markdown material shows up as deleted from the root. This
appears to be a structural reorganization and/or pre-existing worktree state,
not a safe auto-clean target.

Examples:

- `paper2.tex`
- `tau_channel_equations.tex`
- `ceditt4/*.tex`
- multiple `CeDiTT3*` and `cedit_*` manuscript files
- several project notes and handoff notes

Recommended action:
- keep only the manuscripts still active for code development
- for the current state, this means:
  - keep `manuscripts/active/paper2.tex` and `manuscripts/active/paper2.bib`
  - treat `alpha_resonances` as external to this repo
  - treat the rest as inactive / Overleaf-managed material, with no need to
    preserve them further inside this repository

### 4. Untracked research/support files

There are untracked files that may be intentional working material:

- `gaussian_alignment.py`
- `h30h30_resonance.py`
- multiple diagnostic scripts under `scripts/`
- `tmp/`
- `output/`
- small Gaussian input/support files such as `.gjf`, `.chk`, `.xyz`

Recommended action:
- keep and commit those that belong to the active workflow
- archive or ignore generated outputs

## Suggested next commit split

To keep history readable, use separate commits:

1. `feat(gui): integrate shared vibro-rotational workspace`
2. `feat(alpha): add quasiparticle alpha workflow and mixed-source inputs`
3. `refactor(data): make gaussian/ the canonical benchmark location`
4. `docs: align GUI and distribution documentation`
5. `chore(repo): archive legacy app/tmp material`

## Verified after cleanup

- `python3 -m py_compile ceditt_gui.py gaussian_vpt_parser.py distortion_workflow.py`
- selected pytest subset passed after canonical data migration

## Do not auto-clean next

Avoid bulk deletion of these areas without an explicit review:

- tracked release bundles under `dist/`
- any `.app` bundle still intended for distribution
- research scripts under `scripts/`
