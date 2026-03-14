CeDiTT1.0
Centrifugal Distortion Tensor Transformation

Contenuto consigliato della distribuzione
- `CeDiTT1.0.dmg`
- `CeDiTT1.0.app`
- `README_CeDiTT1.0.txt`
- `DOCUMENTAZIONE_CeDiTT1.0.txt`

Installazione consigliata
1. Apri `CeDiTT1.0.dmg`.
2. Trascina `CeDiTT1.0.app` nella cartella `Applications`.
3. Espelli il disco `CeDiTT1.0`.
4. Apri `CeDiTT1.0.app` da `Applications`.

Se macOS blocca il primo avvio
1. Fai clic destro su `CeDiTT1.0.app`.
2. Scegli `Open`.
3. Conferma di nuovo `Open`.

Requisiti
- macOS 11 o successivo
- Apple Silicon (`arm64`)

Note
- La build e standalone e non richiede Python installato sul Mac di destinazione.
- La app non e notarizzata; un avviso di sicurezza al primo avvio e normale.
- Per la documentazione d’uso della GUI vedi `DOCUMENTAZIONE_CeDiTT1.0.txt`.
- Le tab della GUI sono scrollabili verticalmente e anche i report finali hanno lo scroll.
- Se fornisci un file `XYZ`, la app ricava automaticamente gruppo puntuale, `sigma` rotazionale e costanti `A, B, C` da geometria.
- Nei transform manuali quartici/sestici, se e presente un `XYZ`, l’ordine spettroscopico `A,B,C` usato dalla GUI viene armonizzato con quello ottenuto dal file `XYZ`.
- Nei percorsi armonici da `.fchk` oppure `XYZ + Hessian` la GUI mantiene il modello interno del backend, ma confronta e armonizza il reporting rispetto all’ordine `A,B,C` ricavato dall’`XYZ` quando disponibile.
