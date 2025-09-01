# The Bigram Language Model

L'idea centrale del Bigram approach to language modelling è di processare esattamente due caratteri vicini alla volta.<br/>
Funziona solo su queste piccole sottosequenze locali, ignorando di fatto le informazioni di sequenza circostanti presenti nel contesto complessivo del campione (un nome nel nostro caso).

**È un ottimo punto di partenza.**

```python
for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat two char 'sliding-window'
        print(ch1, ch2)
```
`<S> e
e m
m m
m a
a <E>`

This returns `<S>`e, em, mm, ma, a`<E>` for the name emma.

Code tips
```python
for ch1, ch2 in zip(chs, chs[1:]):
```
chs[1:] è la lista spostata di una posizione<br/>
zip() accoppia ogni carattere con quello successivo<br/>

Esempio con "ciao":<br/>
```python
chs = ['<S>', 'c', 'i', 'a', 'o', '<E>']
chs[1:] = ['c', 'i', 'a', 'o', '<E>']

# Le coppie saranno: ('<S>', 'c'), ('c', 'i'), ('i', 'a'), ('a', 'o'), ('o', '<E>')
```

Un modello ora può interpretare che m è probabile che segua e, mentre a è probabile che concluda un nome, e così via,<br/>
**Nota che `<S>` e `<E>` sono 'caratteri speciali'**, che abbiamo aggiunto per denotare l'inizio e la fine di un nome. In altre parole, aggiungiamo molto esplicitamente 'caratteri speciali' per indicare inizio e fine, e facendo così, questo crea coppie di caratteri aggiuntive

Il modo più semplice per derivare statistiche su quali caratteri seguono quali altri nel nostro `names.txt` è semplicemente contare le combinazioni presenti in quel dataset.<br/>
**Per questo abbiamo bisogno di un dizionario:**

```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        bigram = (ch1, ch2) # bigram is the (ch1, ch2) tupel
        b[bigram] = b.get(bigram, 0) + 1 # If tupel count not ex. -> 0 + 1
```

Il dizionario `b` ora contiene gli accumuli / le 'statistiche' delle combinazioni di caratteri nell'intero dataset<br/>
**Diamo un'occhiata**

```python
# b.items() returns tupels like (('<S>', a), 34)
# sorted() would sort items by tupel, not amount
# to sort by amount: lambda function replaces key with value (amount) high->low
sorted(b.items(), key = lambda keyvalue: -keyvalue[1])
```

`b.items()`:
- Restituisce coppie chiave-valore del dizionario
- Esempio: (('`<S>`', 'a'), 34), (('c', 'i'), 12), ecc.
- Ogni tupla ha: (bigramma, conteggio)

`sorted()` normale:
- Di default ordinerebbe per la chiave (il bigramma)
- Non è quello che vogliamo - vogliamo ordinare per frequenza

`key = lambda keyvalue: -keyvalue[1]`:
- keyvalue è ogni tupla (bigramma, conteggio)
- keyvalue[1] estrae il conteggio (secondo elemento)
- Il - davanti inverte l'ordine: da crescente a decrescente
- Quindi ordina dal conteggio più alto al più basso

[(('n', '\<E\>'), 6763),<br/>
(('a', '\<E\>'), 6640),<br/>
(('a', 'n'), 5438),<br/>
(('\<S\>`', 'a'), 4410),<br/>
(('e', '\<E\>'), 3983),<br/>
(('a', 'r'), 3264),<br/>
(('e', 'l'), 3248),<br/>
(('r', 'i'), 3033),<br/>
(('n', 'a'), 2977),<br/>
....<br/>
(('t', 'b'), 1),<br/>
(('z', 'x'), 1)]<br/>

Per esempio, la combinazione più probabile / più spesso incontrata nel nostro `names.txt` è `(('n', '<E>'), 6763)`<br/>
È più conveniente, in realtà, mantenere questa informazione in un array 2D, piuttosto che in questo dizionario.<br/>
In questo modo, possiamo facilmente accedere al numero di occorrenze di una coppia di caratteri mantenendo una struttura uniforme.

Quindi, dovremmo costruire quell'array/matrice 2D in modo tale che:
- le **righe siano i primi caratteri**
- le **colonne siano i secondi caratteri**

All'interno di questa tabella, una cella contiene il numero di occorrenze per riga e colonna.<br/>
Per la gestione e l'elaborazione dei dati, usiamo **PyTorch**.<br/>
Abbiamo lettere e caratteri speciali. Questo crea un array `28 x 28`.

```python
N = torch.zeros((28,28), dtype=torch.int32) # datatype would otherwise be float32 by default

# Problem: We'll have only chars, but below we index using ints -> Need for mapping
chars = sorted(list(set(''.join(words)))) # set(): Throwing out letter duplicates

# A mapping from letter to number
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

# Copied from above, but now modified for mapping
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    # Neat way for two char 'sliding-window'
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 # Increment cell in 2D by 1
```

Code Tips

`chars = sorted(list(set(''.join(words))))`
- `''.join(words)`: unisce tutte le parole in una stringa
- `set()`: rimuove i duplicati
- `sorted(list())`: converte in lista ordinata alfabeticamente

Risultato: `['a', 'b', 'c', ..., 'z']`

```python
stoi = {s:i for i,s in enumerate(chars)}  # string to index
stoi['<S>'] = 26
stoi['<E>'] = 27
```

- Crea un dizionario che mappa ogni carattere a un numero
- Esempio: `{'a': 0, 'b': 1, ..., 'z': 25, '<S>': 26, '<E>': 27}`
- Serve perché gli array si indicizzano con numeri, non caratteri

```python
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]  # converti carattere in indice
        ix2 = stoi[ch2]  # converti carattere in indice
        N[ix1, ix2] += 1  # incrementa la cella corrispondente
```
**Esempio pratico**<br/>
Per il bigramma ('c', 'i'):
- `ix1 = stoi['c'] = 2`
- `ix2 = stoi['i'] = 8` 
- `N[2, 8] += 1` incrementa  la cella alla riga 2, colonna 8

**Risultato finale**<br/>
La matrice N sarà strutturata così:
- **Righe**: primo carattere del bigramma
- **Colonne**: secondo carattere del bigramma
- **Valore nella cella**: quante volte quella coppia appare

Esempio: `N[26, 0]` contiene quante volte `<S>` è seguito da `a`.

## Visualizzazione Bigram Model

Se visualizzassimo N così com'è non si capirebbe niente, usiamo quindi `matplotlib`
```python
plt.imshow(N);
```
![An image from the static](/img/building-makemore/img.png)

**Un po' brutto**<br/>
Costruiamo qualcosa di più carino per capire cosa sta succedendo.
```python
itos = {i:s for s, i in stoi.items()} # Basically reversing stoi element order

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues') # Heatmap basically
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j] # Add text for heat tiles
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis('off');
```
![An image from the static](/img/building-makemore/img_1.png)

## Le impossibilità

Ci sarà una riga e una colonna che rivelano un problema con i nostri 'caratteri speciali' `<S>` e `<E>`.
C'è una colonna per istanze come (a, `<S>`) (penultima) e una riga per tuple come (`<E>`, a) (ultima). **Queste sono combinazioni impossibili**.

> **Questo distorce qualsiasi statistica su cui potremmo basarci**. È un problema così profondo che dovremmo adattare il nostro modello. Sostituiamo i nostri caratteri speciali `<S>` e `<E>` con un solo carattere speciale e comune: `.`

```python
N = torch.zeros((27, 27), dtype=torch.int32) # 28x28 -> 27x27

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0 # Our special character now has position zero
itos = {i:s for s,i in stoi.items()}

# Copied from above, but now modified
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 # Increment cell in 2D by 1

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues') # Heatmap basically
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j] # Add text for heat tiles
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis('off');
```

![An image from the static](/img/building-makemore/img_2.png)

Questo risolve il problema della riga-colonna di prima, dato che `.` infatti ora può comparire davanti o dopo le lettere.<br/>
Nota che anche `..` è legale, tecnicamente. Potremmo infatti produrre un nome vuoto.

## Building Probability Distributions

Seguiremo le probabilità e inizieremo a campionare il modello, riga per riga come mostrato dalla matrice di correlazione.<br/>
Quindi, iniziamo con le tuple della riga contenente `('.', 'a')`.

```python
# Getting the entire zero-th row 
# (a 1D array of '.' and all letters following)
print("Raw first row's combination counts:\n", N[0], "\n")
print(N[0].shape)

# Raw first row's combination counts:
#  tensor([   0, 4410, 1306, 1542, 1690, 1531,  417,  669,  874,  591, 2422, 2963,
#         1572, 2538, 1146,  394,  515,   92, 1639, 2055, 1308,   78,  376,  307,
#          134,  535,  929], dtype=torch.int32) 
# 
# torch.Size([27])
```

Dato che vogliamo campionare, dobbiamo convertire i conteggi grezzi per riga in probabilità.<br/>
Facciamo questo dividendo ogni cella per la somma delle celle della sua riga. In questo modo, otteniamo una distribuzione di probabilità per questa riga.

```python
p = N[0].float() # probability vector (np.array of floats)
p = p / p.sum()  # normalized probability distribution

print("First Row's distribution:\n", p)

# First Row's distribution:
#  tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
#         0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
#         0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290])
```

Code tips

`p = N[0].float():`
- `N[0]` estrae la prima riga della matrice (indice 0)
- `.float()` converte da interi a numeri decimali
- Esempio: se `N[0] = [0, 5, 3, 0, 2, ...] → p = [0.0, 5.0, 3.0, 0.0, 2.0, ...]`

`p = p / p.sum():`
- `p.sum()` calcola la somma totale della riga (es. 0+5+3+0+2+... = 10)
- Divide ogni elemento per questa somma
- Esempio: `[0.0, 5.0, 3.0, 0.0, 2.0, ...] / 10 = [0.0, 0.5, 0.3, 0.0, 0.2, ...]`

Esploriamo il significato di p e vediamo cosa possiamo farci adesso.

```python
# Sampling from these distributions
# Torch.multinomial -> "Give me probability, I'll give you integer"
# We'll use a PyTorch Generator to make things random yet repeatable (deterministic)
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g) # Generate three random numbers [0;1]
p = p / p.sum()  # compact these random numbers into a distribution

# output: [0.6064, 0.3033, 0.0903]
print(p)
```

## Sampling from Probability Distributions

Per campionare la distribuzione (p), possiamo usare `torch.multinomial()`<br/>
Questa funzione prende una distribuzione di probabilità e fornisce un numero di interi campionati con la distribuzione di probabilità data (nel nostro caso [0.6064, 0.3033, 0.0903] = [60%, 30%, 10%]).

```python
# With probability distribution p, create a list of 20 samples
# [replacement: true] means drawing an element doesn't invalidate drawing this element again
torch.multinomial(p, num_samples=20, replacement=True, generator=g)

# We'd expect ~60% of the 20 items to be 0, ~30% to be 1 , ~10% to be 2
# output: [1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
```
> Maggiore è la dimensione del campione, più precisamente la distribuzione può essere approssimata/replicata.

Ora usiamo lo stesso generatore e applichiamo la logica di campionamento al nostro array bidimensionale delle occorrenze. (o piuttosto alla sua prima riga)<br/> 
Prendiamo i conteggi per riga, li comprimiamo in una distribuzione normale ed estraiamo **un** campione da essa.

```python
p = N[0].float() # probability vector
p = p / p.sum()  # normalized probability distributions
# p = [0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
# 0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
# 0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290]
# Indice 0: . → 0.0000 (0%)
# Indice 1: a → 0.1377 (13.77%) ← Probabilità più alta
# ...

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

# This is an index, a number representing a letter by probability
print(itos[ix]) # Convert index to letter

# j
```

Code Tips

`torch.multinomial()` <br/>
La funzione estrae un singolo indice basandosi su queste probabilità:
- Ha maggiori possibilità di estrarre l'indice 1 (a) perché ha probabilità 13.77%
- Ha poche possibilità di estrarre l'indice 0 (.) perché ha probabilità 0%
- Il seed fisso 2147483647 garantisce che l'estrazione sia riproducibile

Abbiamo appena estratto un carattere/token iniziale 'j' per il nostro primo suggerimento per il nome.<br/>
Con quello, possiamo muoverci attraverso l'array per trovare la riga delle voci con nome di riga ('j', '.') e ripetere il processo di estrazione.<br/>
**Da ora in poi questo è un ciclo guidato dalle probabilità.**

```python
g = torch.Generator().manual_seed(2147483647)
n = 20

for i in range(n):
    ix = 0   # Start with special ('.', 'letter') token row
    out = [] # hold the n names to be generated
    while True:
        p = N[ix].float() # probability vector
        p = p / p.sum()   # normalized probability distributions
        # draw a single sample from this distribution, set this as new row index
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        # if we find ourselves back in the special ('.', 'letter') row, we're done with this name
        if ix == 0:
            break
    print(''.join(out))
    
# junide.
# janasah.
# p.
# cony.
# a.
# nn.
# kohin.
# tolian.
# juee.
# ksahnaauranilevias.
# dedainrwieta.
# ssonielylarte.
# faveumerifontume.
# phynslenaruani.
# core.
# yaenon.
# ka.
# jabdinerimikimaynin.
# anaasn.
# ssorionsush.
```
Et voilà, una lista terribile di nomi.

Anche se sta andando malissimo, sta funzionando ragionevolmente.

**Fixiamo il problema**.<br/>
Con `p = N[ix].float() # vettore delle probabilità` stiamo sempre recuperando una riga e convertiamo sempre questa riga interamente da `int` a `float`.<br/>
Inoltre, ad ogni iterazione facciamo anche `p = p / p.sum()`.<br/>
Per questo sarebbe meglio preparare una matrice dedicata e preprocessata `P`; semplicemente una matrice di probabilità calcolate.<br/>
Come passo aggiuntivo, usiamo `P` per sommare se stessa per righe. Questo era `p.sum()` per tutte le lettere prima.<br/>
Costruiamo questa matrice di aggiornamento delle prestazioni `P`:

```python
P = N.float()
# P /= P.sum() # This would sum over all elements, row- and column-wise -> wrong
# This is allowed with PyTorch:
P /= P.sum(1, keepdims=True) # sum: A 27x1 vector (1 stands for row-wise sum) (27 by 27 divided by 27 by 1 is possible in PyTorch -> broadcasting)
# For broadcasting to work like here, each dimension must be either equal or 1 (or not existent), which is the case here (dimensions will be aligned from right to left!)
# Keepdim=True means that the sum vector is 27x1, (the 1 before that stating that the no. of rows is to be kept, but columns are to be summed over per row)

g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    ix = 0
    out = [] # Hold multiple names
    while True:
        p = P[ix]
        # draw a single sample from this distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        # If stopping special character is drawn
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```

```python
# Quick sanity check for broadcasting
# Expected: Every row of P should sum up to 1
print(P.sum(1)) # 1 stands for row-wise sum
```

**Code Tips**<br/>
- `P.sum(1)` calcola la somma lungo la dimensione 1, cioè **somma ogni riga**
- Il risultato mostra che ogni riga di `P` somma esattamente a `1.0`. Ogni riga rappresenta una distribuzione di probabilità quindi i valori in ogni riga devono avere 1 come somma. Il fatto che tutte le riche abbiano come somma `1.0` conferma che la normalizzazione è avvenuta correttamente

**Broadcasting**:
  - Il broadcasting è silenzioso: non ti avvisa se stai facendo qualcosa di sbagliato 
  - Può accettare operazioni non intenzionali: potresti pensare di fare un'operazione, ma PyTorch ne fa un'altra 
  - Gli errori sono difficili da individuare: il codice funziona, ma i risultati sono sbagliati
  ```python
    # Potresti voler fare:
    a = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
    b = torch.tensor([1, 2])            # shape: (2,)
    result = a + b  # Broadcasting: b diventa [[1, 2], [1, 2]]
    
    # Ma forse intendevi:
    b = torch.tensor([[1], [2]])        # shape: (2, 1)  
    result = a + b  # Broadcasting diverso!
  ```
Per saperne di più sul broadcasting: [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

## Qualità dei nomi generati

Abbiamo costruito un bigram language model contando le frequenze delle combinazioni di lettere e poi normalizzando e campionando con quella base di probabilità.<br/>
Abbiamo fatto il train del modello, abbiamo campionato dal modello (in moto iterattivo, carattere per carattere). Ma è ancora scarso nels generare i nomi.<br/>
**Ma quanto è scarso**? Sappiamo che la conoscenza del modello è rappresentata da `P`, ma come possiamo riassumere la qualità del modello in un valore?

Per prima cosa, guardiamo i bigrams creati dal dataset: Il bigram di `emma` è per esempio: `.e, em, mm, ma, a.`

**Che probabilità assegna il modello ad ognuno di questi biagrams?**

```python
# Copied from above, but now modified
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        print(f'{ch1}{ch2}: {prob:.4f}')
        
# Output
# .e: 0.0478
# em: 0.0377
# mm: 0.0253
# ma: 0.3899
# a.: 0.1960
```
Questo codice fornisce degli output come `ma: 0.3899`<br/>
Qualsiasi cosa sopra o sotto $\frac{\partial 1}{\partial 27} = 0.0370$ significa che ci distacchiamo dalla media.<br/>
Abbiamo imparato qualcosa (non è detto che sia costruttivo) sulla bigram statistics.<br/>
L'abbiamo imparato contando le occorrenze di ogni biagram nel dataset dei nomi e successivamente normalizzando i conteggi in probabilità.<br/>

Ora come possiamo riassumere queste probabilità in un indicatore che misuri la qualità?<br/>
Soluzione: La funzione (log) di verosomiglianza (log-likelihood), la somma di $log(probability)$ di tutte le singole probabilità dei token (il logaritmo viene applicato per leggibilità)

> **Più alta è la log-likelihood, migliore è il modello, perchè è più capace di predire il carattere successivo in una sequenza del dataset**

```python
log_likelihood = 0.0
n = 0 # tuple count

# copied from above, but now modified - Log likelihood over all words
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}') # As this is a tensor and we want to see that too
nll = -log_likelihood
print(f'{nll=}')            # Negative log likelihood
print(f'{nll/n}')           # Average negative log likelihood (this is the loss we want to minimize)

# Output
# log_likelihood=tensor(-559891.7500)
# nll=tensor(559891.7500)
# 2.454094171524048
```

Abbiamo calcolata la log-likelihood negativa, perchè questo segue la convenzione di impostare l'obiettivo di minimizzare la funzione di perdita. Più bassa la perdita negativa log-likelihood, migliore è il modello.
> Peggiore/minore è la probabilità, più negativa è la log-likelihood. Questo è il perchè è stata convertita nello spazio positivo. **Più è alta la log-likelihood negativa `nll`, peggiore è il modello**.
> Spesso questa `nll` successivamente viene normalizzata anche esse, diventando la **log-likelihood (negativa) media**.

Noi abbiamo $2.45$ per il modello. Più è bassa meglio è.<br/>
Dobbiamo trovare i parametri che riducano questo valore.

**Goal:**<br/>
Massimizzare la likelihood dei dati trained rispetto ai parametri del modello `P`<br/>
- Questo equivale a: Massimizzare la log-likelihood (dato che il logaritmo è monotono)
- Questo equivale a: Minimizzare la log-likelihood negativa
- Questo equivale a: Minimizzare la media log-likeligood negativa (la misura di qualità, come mostrato in precedenza `2.45`)

I problemi con il nostro modello possono, d'ora in poi, essere visualizzati osservando la nostra funzione di perdita (loss function), vale a dire la log-likelihood negativa.

Uno dei problemi si presenta prontamente qui sotto:
```python
log_likelihood = 0.0
n = 0

# Copied from above, but now modified
for w in ['andrejq']:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(f'\n{log_likelihood=}') # As this is a tensor and we want to see that too
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

# Output
# .a: 0.1377 -1.9829
# an: 0.1605 -1.8296
# nd: 0.0384 -3.2594
# dr: 0.0771 -2.5620
# re: 0.1336 -2.0127
# ej: 0.0027 -5.9171
# jq: 0.0000 -inf
# q.: 0.1029 -2.2736
# 
# log_likelihood=tensor(-inf)
# nll=tensor(inf)
# inf
```
Con il nome `andrejp` abbiamo una media log-likelihood negativa pari a $\infty$. Una perdita infinita, il "worst case" delle performace del modello.<br/>
Questo perchè il bigram `jq` non è mai sato presente nel nostro training data, il conteggio è 0, la likelihood perciò è $0%$<br/>
La likelihood che il modello sceglie è $log(0) = -\infty$

## Model Smoothing

**Model Smoothing** risolve questo abbastanza facilmente.<br/>
Fondamentalmente incrementiamo ogni conteggio che abbiamo di `1` per evitare di avere lo `0`:

```python
P = (N+1).float() # Adding a lot more means smoothing out distributions more; see NN approach for discussing this
# This is allowed with PyTorch:
P /= P.sum(1, keepdims=True) # sum: A 27x1 vector (1 stands for row-wise sum)
```
Aggiungiamo `+1` a `N`, in modo da evitare $\infty$ come risultato del $log$ <br/>
Rieseguendo esattamente lo stesso codice di prima, ora lo smoothing assegna al biagramma `jq` una probabilità (molto piccola).
> Il modello è stato sorpreso da questo biagramma, ma non è più sopraffatto

```python
log_likelihood = 0.0
n = 0

# Copied from above, but now modified
for w in ['andrejq']:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Neat way for two char 'sliding-window'
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
print(f'{log_likelihood=}') # As this is a tensor and we want to see that too
nll = -log_likelihoodF
print(f'{nll=}')
print(f'{nll/n}')

# Output
# log_likelihood=tensor(-27.8672)
# nll=tensor(27.8672)
# 3.4834020137786865
```
**Questo è un modello di stima dei caratteri bigramma abbastanza solido fino a questo punto.**<br/>
**Abbiamo valutato le prestazioni e rimosso i problemi attraverso lo smoothing.** 

**E' ancora un po' traballante.**















