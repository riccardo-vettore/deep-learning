
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

I caretteri della parola sono mappati in un intero e adesso in un vettore di tipo `float32`.<br/>
E questo a sua volta significa che puo' essere un input per una Neural Network.

Facciamo un esperimento con i neuroni. Costruiamo un neurone a 27 dimensioni e lo approcciamo con l'input lettera per lettera del nostro primo nome `.emma`.

```python
W = torch.randn((27,1), generator=g) # the neuron: random column vector of 27 numbers from normal distribution
a = xenc @ W  # '@' is PyTorch's matrix multiplication operator (5x27 @ 27x1 -> 5x1)

print(a) # this is now a 5x1 vector

# output
tensor([[ 0.1066],
        [-1.2464],
        [-0.6378],
        [-0.6378],
        [ 1.8598]])
```
`W` è un nurone **singolo**.

**Code Tips**<br/>

`W = torch.randn((27,1), generator=g)`
- `W`: rappresenta i **pesi** del neurone
- `torch.randn((27,1))`: crea una matrice 27×1 con numeri casuali da distribuzione normale (media=0, deviazione standard=1)
- `generator=g`: usa un generatore specifico per riproducibilità (stesso seed = stessi numeri casuali)

`a = xenc @ W`
- `@`: operatore di moltiplicazione matriciale in PyTorch
- `xenc`: matrice 5×27 (input one-hot di ".emma")
- `W`: matrice 27×1 (pesi del neurone)

```text
xenc (5×27)  @  W (27×1)  =  a (5×1)
[1,0,0,...,0]   [w₁]        [w₁]     # per '.'
[0,0,0,1,0,.]   [w₂]        [w₅]     # per 'e'  
[0,0,...,1,.]   [w₃]    =   [w₁₃]    # per 'm'
[0,0,...,1,.]   [...]       [w₁₃]    # per 'm'
[0,1,0,...,0]   [w₂₇]       [w₂]     # per 'a'

Elemento (0,0): riga 0 di xenc x colonna 0 di W
[1,0,0,...,0] • [w₁, ..., w₂₇] = w₁ x 1 + 0 x w₂ ... + 0 x w₂₇ = w₁
Elemento (1,0): riga 1 di xenc x colonna 0 di W
[0,0,0,1,0,.] • [w₁, ..., w₂₇] = w₁ x 0 + 0 x w₂ + ... 1 x w₅ ... + 0 x w₂₇ = w₅
...
```

Il vettore mostra l'attivazione del neurone per carattere; in altre parole la sua reazione ai caratteri.
> Poichè l'input era codifica one-hot, il singolo neurone riceve un "carattere della lunghezza di 27 dimensioni".
> Per questo esempio lo fa per tutte le lettere contemporaneamente. Per ogni lettera a 27 dimensioni, produce esattamente un'attivazione. Non impara da questo, ma questa è l'idea generale.<br/>
> L'intuizione chiave è che **siccome una lettera ha 27 dimensioni attraverso la Codifica One-Hot, anche un singolo neurone deve avere 27 dimensioni.**

Questo era solo **un** neurone. Noi vogliamo 27 neuroni.<br/>
La ragione per "un neurone per ogni possibile carattere" la vedremo successivamente.

```python
W = torch.randn((27,27), generator=g) # random column matrix of 27x27 numbers (previous was 27x1 for a single neuron)
a = xenc @ W  # @ is PyTorch's matrix multiplication operator, this is now a 5x27 vector

print(a) # this is now a 5x27 vector
```
tensor([[ 0.2603,  0.9090, -1.4458,  1.1072, -0.7175, -0.3867, -1.2542,  1.2068,
-0.7305, -1.0926,  0.3223,  0.0717, -0.2774,  1.1634, -0.6691,  0.6492,
-0.8157,  0.6404,  1.0442, -1.1571,  0.5107,  0.7593, -1.6086, -0.1607,
-0.7226,  0.5205,  0.7270],
[ 0.9641,  0.0471,  0.3096,  1.2087, -0.9954, -0.4485, -1.2345,  1.1220,
-0.6738,  0.6365, -0.5964,  1.3058,  0.3857, -0.7510,  0.9278, -1.4849,
-0.2129, -0.9419,  1.5729,  1.0105, -0.1085,  0.6006, -0.7091,  1.9217,
-0.1818, -0.0954, -0.9253],
[-0.4645, -0.5206, -0.5579,  1.1087,  0.4149,  0.9557, -0.1471, -1.2532,
-1.1850,  2.1940,  0.6698,  0.4829,  2.0022, -0.6284, -0.9379,  1.6772,
0.0039, -0.1460, -1.2915, -0.0748,  1.3272,  1.6676,  1.3931,  0.6540,
-0.2245, -1.8563,  0.9609],
[-0.4645, -0.5206, -0.5579,  1.1087,  0.4149,  0.9557, -0.1471, -1.2532,
-1.1850,  2.1940,  0.6698,  0.4829,  2.0022, -0.6284, -0.9379,  1.6772,
0.0039, -0.1460, -1.2915, -0.0748,  1.3272,  1.6676,  1.3931,  0.6540,
-0.2245, -1.8563,  0.9609],
[ 0.1114, -0.5977, -0.3977, -1.2801,  0.0924, -0.1463, -0.5254, -1.5195,
0.3240, -1.5065,  1.2898, -1.5100,  1.0930,  0.0549,  1.3537, -1.0896,
0.2558,  0.2469,  0.3190, -0.9861, -0.2138, -3.0010,  1.4111,  0.0317,
-0.5475,  0.8183, -0.8163]])

Questo valuterà in parallelo tutti i $27$ neuroni su tutti gli esempi.<br/>
L'output è ora una matrice $5 x 27$.<br/>
**Questo significa**: Per ognuno dei 27 neuroni, otteniamo il tasso di attivazione del neurone su ognuno dei 5 esempi.

```python
(xenc @ W)[3, 13] # The firing rate of the 14th neuron at the 4th input

# Output
# tensor(-0.6284)
```

Ora abbiamo alimentato 5 input a 27 dimensioni in un layer di input di 27 neuroni.<br/>
**Non aggiungeremo un Bias o altro**. Questo è tutto dal lato della struttura della rete.

## Ripristino delle distribuzioni normali
Per ogni carattere di input, vogliamo che i neuroni producano 27 numeri.<br/>
Questi 27 numeri dovrebbero formare una **distribuzione di probabilità normale** che ci dica quanto è probabile ogni possibile carattere successivo.<br/>
Come abbiamo visto nel modello Bigram, dove avevamo probabilità chiare per ogni transizione carattere → carattere

**Problema**: Al momento non abbiamo questo.<br/>
Dalla rete neurale otteniamo 27 numeri per carattere, ma sono valori grezzi (raw values). Possono essere positivi, negativi, di qualsiasi grandezza. <br/>
**Non sono probabilità** (dovrebbero essere tra 0 e 1 e sommati devono valere 1).

Le reti neurali non producono automaticamente distribuzioni di probabilità. L'output grezzo potrebbe essere qualcosa come:
```python
[-2.3, 0.7, 15.2, -0.1, 8.9, ...]
```
Ma noi abbiamo qualcosa come:
```python
[0.05, 0.12, 0.03, 0.45, 0.08, ...] // somma = 1.0
```

**Soluzione**<br/>
Nel modello Biagram avevamo una matrice di conteggi diretti:
```python
    a  b  c  d
a [ 5, 2, 0, 1]  // dopo 'a', abbiamo visto: 5 volte 'a', 2 volte 'b', etc.
b [ 1, 3, 4, 0]
c [ 0, 1, 2, 6]
d [ 2, 0, 1, 4]
```
Da questi conteggi calcolavamo facilmente la probabilità dividendo il totale.

**Soluzione: Interpretazione come log-counts**<br/>
L'idea chiave è: interpretiamo questi numeri come **logaritmi** dei conteggi originali.<br/>
Se la rete producer `[-2.3, 0.7, -∞, 0.0]` <br/>
Li interpretiamo come:
- log(conteggio_a) = -2.3
- log(conteggio_b) = 0.7
- log(conteggio_c) = -∞
- log(conteggio_d) = 0.0

**Recuperiamo i "conteggi" con l'esponenziale** <br/>
Applichiamo $e^x$ per ottenere i conteggi
- $e^{-2.3} = 0.1$
- $e^{0.7} = 2$
- $e^{-∞} = 0$
- $e^{0.0} = 1.0$
Risultato: `[0.1, 2.0, 0, 1.0]` sembrano conteggi realistici.<br/>
Questo è essenzialmente quello che fa la funzione softmax internamente!RetryClaude does not have the ability to run the code it generates yet.

```python
logits = xenc @ W # logits, different word for log-counts

# These two combined are called Softmax -> Build a probability distribution from logits
counts = logits.exp() # negative numbers are positive below 1, positive numbers are positive above 1
# Let's just say the counts variable holds something like 'fake counts', kinda like in the N matrix of bigram, we process them just the same
probs = counts / counts.sum(1, keepdims=True) # Normal distribution probabilities

print(probs.shape)    # 5x27, as expected
print(probs[0].sum()) # Will be 1. for any index [0-4]
```

## Recap
Dato l'input di esempio `.emma`, la rete neurale elabora un carattere alla volta. Iniziamo con l'input `x = .` e la label `y = e` e cosi via.
- Otteniamo l'indice di `.` che è `0`
- Codifichiamo il `.` in one-hot basandoci sull'indice `0` per formare un vettore 27-dimensionale
- Questo è entrato nella rete neurale come vettore
- Li ha attivato diversi neuroni 27-dimensionali
- Le attivazioni per il `.` formano quindi una matrice
- Poi viene applicata la Softmax:
  - Le attivazioni/logit vengono elaborate attraverso $e^x$ per portarle nel range corretto $(0, ∞)$
  - Questi 27 valori logit spostati vengono sostituiti nuovamente calcolando le loro probabilità normali combinate

Pensiamo alla Softmax come una funzione di normalizzazione che prende numeri 'strani' e restituisce una distribuzione normale positiva. Queste probabilità normali devono indicare quale lettere deve sequire, ad esempio l'input `.`.

**La domanda ora è:**<br/>
Possiamo trovare un set di Weights `W` in modo tale che le probabilità che la rete setta siano buone?

**Le operazione elencate prima/dopo sono differenziabili e quindi backpropagatable**

Per completenza, riscriviamo
```python
# FORWARD-PASS:
xenc = F.one_hot(xs, num_classes=27).float() # one-hot encode the names
logits = xenc @ W # logits, different word for log-counts
# Softmax as part of forward pass
counts = logits.exp() # 'fake counts', kinda like in  the N matrix of bigram
probs = counts / counts.sum(1, keepdims=True) # Normal distribution probabilities

print(probs.shape)

# Output
# torch.Size([5, 27])
```


