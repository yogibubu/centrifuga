# CentrifugalTransform - Documentazione

## 1. Scopo
`CentrifugalTransform` e un programma Python con GUI per trasformare costanti di distorsione centrifuga tra rappresentazioni assiali di Watson (`I`, `II`, `III`) e riduzioni (`A`, `S`).

Il progetto include:
- trasformazioni quartiche (riferimento Yamada e metodo tensoriale)
- trasformazioni sestiche (metodo tensoriale)
- diagnostica numerica di stabilita (condition numbers)
- build in applicazione macOS (`.app`)

## 2. File principali
- `ceditt_gui.py`: programma principale con interfaccia grafica
- `build_app.sh`: script di build dell'app macOS
- `CentrifugalTransform.app`: app compilata
- `test_tensor_algorithm.py`: test quartico (tensoriale vs riferimento)
- `test_tensor_sextic_algorithm.py`: test sestico (round-trip)

## 3. Requisiti
- Python 3
- `numpy`
- `tkinter` (normalmente incluso su macOS con Python di sistema/conda)

## 4. Avvio del programma
Da terminale:
```bash
cd /Users/vincenzobarone/centrifugal
python3 ceditt_gui.py
```

## 5. Struttura GUI
La GUI ha 2 tab:

### 5.1 Tab Quartic
Input:
- costanti rotazionali `A, B, C` (MHz)
- rappresentazione input (`I/II/III`)
- riduzione (`A` o `S`)
- costanti quartiche input (kHz)

Metodi disponibili:
- `Reference (Yamada / Yamada-like S)`
- `Tensor (quartic)`

Output:
- costanti trasformate nelle altre 2 rappresentazioni
- report di stabilita numerica

Diagnostica mostrata:
- `cond2`, `cond1`, `condinf`
- `sigma_min`, `sigma_max`
- `cond2 * eps`
- gap rotazionali: `A-B`, `B-C`, `A-C`

### 5.2 Tab Sextic
Input:
- `A, B, C` (MHz)
- rappresentazione input (`I/II/III`)
- riduzione input e output (`A`/`S`)
- costanti sestiche input (kHz)

Output:
- costanti trasformate nelle altre 2 rappresentazioni
- controllo round-trip numerico

## 6. Convenzioni implementate

### 6.1 Quartiche
- Trasformazione di rappresentazione in percorso `Reference` secondo formalismo Yamada.
- Per riduzione `A`, viene applicata la convenzione standard Watson sul segno di `deltaK`:
  - dopo ogni trasformazione: `deltaK <- -deltaK`

### 6.2 Sestiche
- Trasformazione effettuata via tensore `phi` con pseudoinversa e permutazione assi.
- Cambio riduzione gestito tramite proiezione tensoriale (nessuna formula diretta A<->S).

## 7. Algoritmi (sintesi)

### 7.1 Quartic Tensor
1. Ricostruzione `tau` da Watson input (via `pinv`)
2. Permutazione assi (`I/II/III`)
3. Ricostruzione Watson in output

### 7.2 Sextic Tensor
1. Ricostruzione `phi` da Watson input (via `pinv`)
2. Permutazione assi
3. Proiezione in riduzione/rappresentazione target

## 8. Build applicazione macOS
Build locale:
```bash
cd /Users/vincenzobarone/centrifugal
bash build_app.sh
```

Output:
- `/Users/vincenzobarone/centrifugal/CentrifugalTransform.app`

## 9. Aggiornare l'app sul Desktop
```bash
cp -R /Users/vincenzobarone/centrifugal/CentrifugalTransform.app /Users/vincenzobarone/Desktop/CentrifugalTransform.app
```

## 10. Validazione raccomandata
- confronto `Reference` vs `Tensor` in quartica su stesso input
- round-trip rappresentazione (`I -> III -> I`)
- round-trip riduzione (`A -> S -> A`) nei casi di test
- monitoraggio condition number prima di interpretare parametri sensibili (es. `deltaK`)

## 11. Note pratiche
- Un `A-B` piccolo peggiora il condizionamento della trasformazione.
- In questi casi, piccole incertezze di input possono amplificarsi molto su alcune costanti quartiche.
