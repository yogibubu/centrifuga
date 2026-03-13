# CeDiTT Report

## Stato del lavoro

Il primo lavoro ha ormai una struttura abbastanza chiara. Il suo oggetto non e'
piu' una teoria generale VPT4, ma la riformulazione tensoriale delle quartiche
standard e delle sestiche standard, con trasporto tra rappresentazioni e
riduzioni ottenuto mediante lifting pseudoinverso e gauge fixing.

Il risultato principale raggiunto finora e' che le quartiche standard e le
sestiche standard possono essere espresse nello stesso linguaggio tensoriale di
base. In particolare, la derivata seconda del tensore d'inerzia non e' piu'
necessaria come oggetto primitivo nel settore standard. La struttura si chiude
sul tensore di primo livello

`mu1 = d(I^-1)/dQ`,

sui couplings di Coriolis e, nel caso sestico, sulle costanti cubiche `phi3`.

## Quartiche e sestiche standard

Per le quartiche standard, il contenuto tensoriale naturale e' governato da
`mu1`. Per le sestiche standard, il lavoro fatto nel codice mostra che l'intero
settore geometrico attualmente implementato e' ricostruibile dagli stessi
oggetti di primo livello: `mu1`, Coriolis, costanti rotazionali e kernel in
frequenza. La parte anarmonica esplicita entra tramite le costanti cubiche
`phi3`.

Questo significa che, nel settore standard, quartiche e sestiche appartengono a
uno stesso settore lineare di trasformazione. In altre parole, possono essere
trattate come rappresentanti tensoriali che si trasformano linearmente sotto
cambio di rappresentazione, una volta fissato il gauge con la pseudoinversa.

## Ruolo di H22

Il primo termine che rompe questo schema e' il canale quartico `H22`.

Qui il tensore di secondo ordine

`mu2 = d^2(I^-1)/dQdQ`

si decompone come

`mu2 = B(mu1) - N`,

dove `B(mu1)` e' la chiusura bilineare interamente generata dal primo livello,
mentre `N` e' il primo nuovo oggetto tensoriale intrinseco.

Questa decomposizione mostra che `H22` e' il primo contributo quartico che non
resta interamente nel settore lineare standard: la parte bilineare appartiene
ancora al linguaggio di `mu1`, mentre la parte `N` introduce un nuovo livello
tensoriale.

## Conseguenza per la struttura del lavoro

Il primo paper puo' quindi essere impostato come segue:

- quartiche standard;
- sestiche standard;
- dimostrazione che entrambe si esprimono nello stesso settore tensoriale
  lineare, eliminando la derivata seconda del tensore d'inerzia come oggetto
  primitivo;
- introduzione di `H22` come primo termine che rompe questo schema.

In questa lettura, `H22` non e' ancora il cuore del VPT4 completo, ma il primo
breaking term del settore standard.

## Cosa manca

Il passo successivo naturale e' sviluppare le sestiche oltre il settore standard
per identificare il primo termine che gioca un ruolo analogo a `H22`, cioe' il
primo termine sestico che obbliga a introdurre un nuovo oggetto tensoriale oltre
il livello `mu1`.

La speranza e' che questo primo termine sia ancora semplice e possa essere usato
come diagnostico strutturale, accanto o oltre `s111`, nello stesso modo in cui
`H22` lo e' sul lato quartico.

## Stato attuale in una frase

Il settore standard di quartiche e sestiche e' ormai chiarito in termini di un
lessico tensoriale comune; `H22` e' stato identificato come primo termine che
rompe questa linearita'; resta da trovare il primo breaking term sestico.
