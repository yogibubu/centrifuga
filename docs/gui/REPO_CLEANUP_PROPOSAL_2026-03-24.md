# Repo Cleanup Proposal - 2026-03-24

## Why

The repository now mixes several different roles in the top-level tree:

- production/backend code,
- GUI/app code,
- manuscript material,
- snapshots and handoff notes,
- release/distribution artifacts,
- temporary merge helpers.

This is still workable, but it is starting to make navigation and release
management harder than necessary.

## Recommended cleanup

### 1. Separate long-lived source from release artifacts

Keep in the repo root only source and documentation that belongs to active
development.

Candidates to move under a dedicated `release/` or `distribution/` folder:

- `CeDiTT1.0.spec`
- `CeDiTT.spec`
- `build_app.sh`
- app-facing readme/instructions files

Candidates to keep out of git or at least out of the root:

- `dist/`
- `build/`
- generated `.app` bundles
- generated `.dmg` packages

### 2. Consolidate GUI documentation

Current GUI-facing documentation is split across multiple overlapping files.

Candidates to consolidate:

- `GUI_APP_GUIDE.md`
- `CEDITT_GUI_HANDOFF_2026-03-14.md`
- distribution-side readmes

Recommended split:

- one stable user guide
- one developer handoff / changelog
- one release note per external distribution

### 3. Move manuscript-related material into a dedicated subtree

The manuscript and paper-adjacent notes should not be visually mixed with app
and backend entrypoints.

Recommended structure:

- `manuscript/`
- `manuscript/notes/`
- `manuscript/snapshots/`

This would clarify the boundary between:

- code that computes rovibrational quantities,
- text that interprets or reports them.

### 4. Isolate temporary and merge helpers

These are useful, but they should not sit at the same level as production
modules.

Candidates:

- `tmp_merge/`
- ad hoc handoff snapshots
- one-off diagnostic notes that are no longer active

Recommended structure:

- `archive/`
- `archive/tmp_merge/`
- `archive/handoffs/`

### 5. Group app-specific code explicitly

At the moment the app code is still easy to find, but it is not visually
separated from the broader backend.

Possible future split:

- `app/ceditt_gui.py`
- `app/assets/`
- `app/release/`

while keeping the scientific backend modules in the root or in a dedicated
`backend/` package.

### 6. Clarify Gaussian benchmark data location

There are overlapping `.log` / `.fchk` files both in the repo root and in
`gaussian/`.

Recommended cleanup:

- keep one canonical benchmark dataset location
- move ad hoc duplicates out of the root
- document which location the scripts should treat as authoritative

## Low-risk first pass

The safest immediate cleanup pass would be:

1. declare one canonical benchmark data directory
2. consolidate GUI docs
3. move temporary merge helpers and stale handoff artifacts into `archive/`
4. keep release artifacts grouped together

## Deliberately not done here

This note does not move or delete files.

Reason:

- the repository contains manuscript snapshots, app bundles, and handoff
  material that may still be serving as safety copies;
- any destructive cleanup should be done only after you choose which artifacts
  are canonical and which are archival.
