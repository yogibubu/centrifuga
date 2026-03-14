# CeDiTT GUI Handoff - 2026-03-14

## Stato attuale
- La GUI parte correttamente su macOS senza il crash iniziale della menubar.
- Le tab `Quartic` e `Sextic` sono scrollabili.
- I report `Quartic` e `Sextic` sono scrollabili.
- Il parser `XYZ` accetta anche file con numeri atomici nella prima colonna.
- Se e presente un `XYZ reference`, la GUI ricava:
  - gruppo puntuale
  - `sigma`
  - `rotor class`
  - `A, B, C` da geometria

## Armonizzazione XYZ / ABC
- Nei transform manuali quartici e sestici:
  - i dati manuali restano interpretati nella `representation` e `reduction` selezionate;
  - l'`XYZ reference` fissa l'ordine spettroscopico `A,B,C`;
  - e disponibile il pulsante `Use XYZ A,B,C`.
- Nei percorsi armonici (`.fchk` oppure `XYZ + Hessian`):
  - il backend mantiene il proprio modello interno;
  - il report confronta `ABC_model` e `ABC_xyz`;
  - viene stampato uno stato `OK/CHECK/WARNING`.

## Stato fisico dei rami H22
- Quartico `H22`:
  - trattato come ramo validato;
  - benchmark CeDiTT3 riprodotto dal path dedicato.
- Sestico `H22`-linear:
  - lasciato esplicitamente come diagnostic candidate;
  - non presentato come benchmark quantitativo assoluto allo stesso livello del quartico `H22`.

## Verifiche eseguite
- `python3 -m pytest -q test_ceditt_gui_xyz_harmonization.py test_h22_reference_path.py test_h22_decomposition.py test_harmonic_convention.py`
  - `9 passed`
- Verifica precedente piu ampia GUI/backend:
  - `python3 -m pytest -q test_ceditt_gui_xyz_harmonization.py test_harmonic_convention.py test_h30h30_structure.py test_h30h30_resonance.py`
  - `19 passed`
- Test pratico con:
  - [h2s.xyz](/Users/vincenzobarone/centrifugal/h2s.xyz)
  - [h2s.fchk](/Users/vincenzobarone/centrifugal/h2s.fchk)
  - [h2s.log](/Users/vincenzobarone/centrifugal/h2s.log)

## File principali
- [ceditt_gui.py](/Users/vincenzobarone/centrifugal/ceditt_gui.py)
- [rovib_distortion.py](/Users/vincenzobarone/centrifugal/rovib_distortion.py)
- [test_ceditt_gui_xyz_harmonization.py](/Users/vincenzobarone/centrifugal/test_ceditt_gui_xyz_harmonization.py)
- [README_CeDiTT.txt](/Users/vincenzobarone/centrifugal/README_CeDiTT.txt)
- [DOCUMENTAZIONE_CeDiTT1.0.txt](/Users/vincenzobarone/centrifugal/DOCUMENTAZIONE_CeDiTT1.0.txt)
- [GUI_APP_GUIDE.md](/Users/vincenzobarone/centrifugal/GUI_APP_GUIDE.md)
- [dist/CeDiTT1.0.app](/Users/vincenzobarone/centrifugal/dist/CeDiTT1.0.app)

## Snapshot di sicurezza
- Snapshot sorgenti:
  - [source_snapshots/20260314_111812](/Users/vincenzobarone/centrifugal/source_snapshots/20260314_111812)

## Prossimo passo consigliato
- Fare un commit dedicato dei fix GUI/documentazione/build, oppure copiare di nuovo la `.app` aggiornata nella distribuzione Desktop se vuoi allineare anche la bundle esterna all'ultimo stato.
