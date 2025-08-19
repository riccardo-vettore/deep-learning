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

**Questo distorce qualsiasi statistica su cui ci baseremmo**. È un problema così profondo che dovremmo adattare il nostro modello. Questo viene fatto sostituendo i nostri caratteri speciali `<S>` e `<E>` con un solo carattere speciale e comune: `.`

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




