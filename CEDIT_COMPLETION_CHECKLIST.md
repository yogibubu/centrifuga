# CeDiTT Completion Checklist

This checklist is for **Project 1 only**:

- standard quartic `H12H12`
- quartic representation transforms
- analogous sextic transform framework
- CeDiTT app

It explicitly excludes the full VPT4 workflow.

## 1. Quartic H12H12 structure

- [x] validated quartic order-2 route against Gaussian/QCent
- [x] corrected pseudoinverse transform used in the GUI
- [x] Gaussian-consistent projection `tau -> Tau' -> T -> Watson`
- [x] pair-pair closure documented
- [x] `Tau'` spectral invariants exposed in the workflow
- [x] full `3+3` spectral parameterization of `Tau'` exposed in the workflow
- [ ] decide and freeze the final interpretation of the 3 spatial invariants
- [ ] identify the 2 residual coordinates needed for the intended `3+2` structure as a reduction/gauge choice on top of the `3+3` description

## 2. Sextic transform side

- [x] sextic tensor transform in the app validated on the modeled physical subspace
- [x] Gaussian sextic benchmark replicated numerically for `h2o`
- [x] app diagnostics for physical-subspace residual
- [ ] write a compact mathematical note tying the sextic app transform to the quartic tensor logic

## 3. CeDiTT app

- [x] quartic tensor route is the default path
- [x] legacy Yamada check moved to a separate button
- [x] sextic transform corrected
- [x] app rebuilt and redistributed
- [ ] final GUI wording review for quartic spatial/gauge language
- [ ] optional export of `Tau'` invariants / gauge diagnostics in the app

## 4. Documentation

- [x] README updated to distinguish Project 1 and Project 2
- [x] app README updated
- [ ] short internal note: what belongs to CeDiTT, what belongs to the full workflow
- [ ] final pass on code comments so that `H12H12/CeDiTT` diagnostics are not read as VPT4-global statements

## 5. Stop condition for CeDiTT

CeDiTT can be considered closed when:

1. the quartic transform route is frozen,
2. the sextic transform route is frozen,
3. the app UI/README are consistent,
4. the spatial/gauge interpretation used in the app/manuscript is fixed at the `H12H12` level.
