
# Neural Network Approach - Lo stesso problema, differente soluzione

**Ora**, formuleremo il problema della stima dei caratteri nel framework delle Neural Networks.<br/>
Il problema come l'abbiamo affrontato rimane il medesimo cambia però l'approccio, il risultato dovrebbe apparire simile.

La nostra neural network **riceve un singolo carattere** e **produce la distribuzione di probabilità sui possibili caratteri successivi** (27 in questo caso).<br/>

Farà delle supposizioni sul carattere più probabile che segue.<br/>
Questo sarà ancora misurabile nelle prestazioni attraverso la stessa funzione di loss, la log-likelihood negativa.<br/>
Dato che abbiamo i dati di addestramento, per ogni esempio di addestramento, conosciamo il carattere che effetivamente viene dopo.<br/>
Questo può essere utilizzato per mettere a punto la neural network per fare supposizioni migliori.<br/>
**Supervised Learning in azione.**

```python
#Create training set of all bigrams
xs, ys = [], [] # Input and output character indices

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# Convert lists to tensors
xs = torch.tensor(xs)
ys = torch.tensor(ys) 
```

With name `.emma.`, `xs` and `ys` would look like this:
```python
#Create training set of one particular bigram
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(f'{ch1}{ch2}: {ix1} -> {ix2}')
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs)
print(ys)

# Output
# .e: 0 -> 5
# em: 5 -> 13
# mm: 13 -> 13
# ma: 13 -> 1
# a.: 1 -> 0
# tensor([ 0,  5, 13, 13,  1])
# tensor([ 5, 13, 13,  1,  0])
```

> Esistono `torch.tensor` e `torch.Tensor`. Quale dovrebbe essere usato?

Ogni PyTorch Tensor è un'istanza di torch.Tensor, ma torch.tensor è una funzione che costruisce e restituisce un'istanza di `tourch.Tensor`.<br/>
Ad eccezione di quando si inizializza un tensor completamente vuoto, in generale non c'è ragione di scegliere `torch.Tensor` rispetto a `torch.tensor`.<br/>
Va notato inoltre che torch.Tensor è un alias di torch.FloatTensor, il defualt `dtype` perciò sarebbe `torch.float32`.<br/>

**Si consiglia l'uso di `torch.tensor`.**

## Feeding the Network
Non ha senso utilizzare le rappresentazioni numeriche delle lettere come input per un singolo neurone di input.<br/>
La rete si riferirebbe al valore numerico stesso, senza alcun riferimento al contesto potenziale come il possibile range di valori.<br/>
In altre parole, rappresentando come a=1, b=2, .., z=26, la rete neurale tratta questi numeri come valori matemateci assoluti con significato quantitativo.<br/>
La rete "pensa":
- `a` (1) è "più piccolo" di `b` (2)
- `z` (26) è "26 volte più grande" di `a` (1)
- La distanza tra `a` e `c` è $2$, mentre `a` e `z` è 25
Alle rete inoltre manca il contesto, non sa che:
- Questi numeri rappresentano categorie discrete (lettere)
- Il range va da 1 a 26 (potrebbe essere da 1 a 100 per quanto ne sa)
- Non c'è una vera relazione ordinale tra le lettere
- `a` e `z` non sono "opposti" matematicamente

La lettera associata a un numero si trova nella sua posizione indicizzata a causa della sua relazione con altre lettere.
Tuttavia questo non verrebbe espresso.<br/>
La **One-Hot Encoding** sarebbe molto migliore per questo scopo.
> Con la One-Hot Encoding prendiamo il valore intero di una lettera, per es $13$ e creiamo un vettore con tutti $0$, ad eccezione per la posizione $13^{\\text{th}}$, nella quale posizioniamo un $1$.

```python
xenc = F.one_hot(xs, num_classes=27).float() # num_classes removes the need for F's guessing
xenc.shape # For '.emma' this will be [5, 27]
plt.imshow(xenc); # '. e m m a' (remember, this is the input, output would be 'e m m a .' for this example)
```

**Code Tips**
`F.one_hot(xs, num_classes=27).float()`
- `F.one_hot()`: funzione PyTorch che converte ogni indice in un vettore one-hot
- `num_classes=27`: specifica che il vocabolario ha 27 classi (26 lettere + 1 il carattere speciale `.`)
- `.float()`: converte il risultato da int a float

Esempio:
```python
0 → [1, 0, 0, 0, ..., 0]  # punto
5 → [0, 0, 0, 0, 0, 1, 0, ..., 0]  # 'e'
13 → [0, 0, ..., 1, 0, ..., 0]  # 'm'
```
- `xenc.shape`: il risultato è una matrice dove:
  - 5 = lunghezza della sequenza (".emma")
  - 27 = dimensione del vocabolario
- `plt.imshow(xenc)`: visualizza la matrice come un'immagine, dove ogni riga rappresenta un carattere e le colonne rappresentano il vocabolario.




