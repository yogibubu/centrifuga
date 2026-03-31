# CeDiTT GUI Handoff - 2026-03-14

## Stato attuale
- La GUI parte correttamente su macOS senza il crash iniziale della menubar.
- E presente una tab `Vibro-Rotational` che usa un dataset condiviso per quartico, alpha e sestico.
- Le tab `Quartic` e `Sextic` sono scrollabili.
- Anche la tab integrata e scrollabile.
- I report `Quartic` e `Sextic` sono scrollabili.
- Il report integrato vibro-rotazionale e scrollabile.
- Il parser `XYZ` accetta anche file con numeri atomici nella prima colonna.
- Se e presente un `XYZ reference`, la GUI ricava:
  - gruppo puntuale
  - `sigma`
  - `rotor class`
  - `A, B, C` da geometria

## Dataset condiviso
- Nella parte alta della finestra e disponibile un blocco `Shared vibro-rotational input`.
- I campi condivisi sono:
  - `.fchk`
  - `XYZ`
  - `Hessian`
  - `anharmonic log`
- Le route armoniche di quartico, alpha e sestico usano questi path come default se i campi locali sono vuoti.
- Questo permette di leggere:
  - quartiche standard
  - diagnostico H22
  - alpha / Delta_vib
  - gerarchia sestica
  come proprieta diverse dello stesso input vibro-rotazionale.

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

## Stato fisico del ramo alpha
- `alpha` e ora coordinato esplicitamente con la lettura vibro-rotazionale della GUI.
- Il report distingue:
  - `Coriolis`
  - `Inertia`
  - `Anharm`
- Il termine `Anharm` e ulteriormente separato in:
  - `phi_iii` come anarmonicita diagonale
  - `phi_iij` come accoppiamento semi-diagonale tra modi
- L'obiettivo e rendere leggibili insieme:
  - contributi ai singoli `alpha`
  - contributi a `Delta_vib`
  - relazione con quartico e sestico sullo stesso dataset

## Verifiche eseguite
- `python3 -m pytest -q test_ceditt_gui_xyz_harmonization.py test_h22_reference_path.py test_h22_decomposition.py test_harmonic_convention.py`
  - `9 passed`
- Verifica precedente piu ampia GUI/backend:
  - `python3 -m pytest -q test_ceditt_gui_xyz_harmonization.py test_harmonic_convention.py test_h30h30_structure.py test_h30h30_resonance.py`
  - `19 passed`
- Test pratico con:
  - [h2s.xyz](/Users/vincenzobarone/centrifugal/data/gaussian/linear_cases/h2s.xyz)
  - [h2s.fchk](/Users/vincenzobarone/centrifugal/h2s.fchk)
  - [h2s.log](/Users/vincenzobarone/centrifugal/h2s.log)
- Build release eseguito con:
  - [build_app.sh](/Users/vincenzobarone/centrifugal/build_app.sh)
  - output aggiornato in [dist/CeDiTT1.0.app](/Users/vincenzobarone/centrifugal/dist/CeDiTT1.0.app)

## File principali
- [ceditt_gui.py](/Users/vincenzobarone/centrifugal/ceditt_gui.py)
- [rovib_distortion.py](/Users/vincenzobarone/centrifugal/rovib_distortion.py)
- [test_ceditt_gui_xyz_harmonization.py](/Users/vincenzobarone/centrifugal/tests/vanvleck_octic/test_ceditt_gui_xyz_harmonization.py)
- [README_CeDiTT.txt](/Users/vincenzobarone/centrifugal/README_CeDiTT.txt)
- [DOCUMENTAZIONE_CeDiTT1.0.txt](/Users/vincenzobarone/centrifugal/DOCUMENTAZIONE_CeDiTT1.0.txt)
- [GUI_APP_GUIDE.md](/Users/vincenzobarone/centrifugal/docs/gui/GUI_APP_GUIDE.md)
- [dist/CeDiTT1.0.app](/Users/vincenzobarone/centrifugal/dist/CeDiTT1.0.app)

## Snapshot di sicurezza
- Snapshot sorgenti:
  - [source_snapshots/20260314_111812](/Users/vincenzobarone/centrifugal/source_snapshots/20260314_111812)

## Prossimo passo consigliato
- Fare un commit dedicato della GUI vibro-rotazionale integrata e della documentazione aggiornata.
- Valutare una pulizia del repo:
  - consolidare documentazione ridondante
  - separare meglio `app`, `manuscript`, `snapshots`, `distribution`
  - spostare file temporanei o distributivi fuori dal root del progetto.
