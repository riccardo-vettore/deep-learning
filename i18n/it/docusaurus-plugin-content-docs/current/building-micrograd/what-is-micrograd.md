# Cos'è Micrograd?

**I motori autograd sono il componente principale che abilita l'addestramento delle reti neurali.**
Micrograd è un piccolo motore autograd che supporta la differenziazione automatica così come il calcolo di gradienti di ordine superiore.
Composto da circa 150 righe di codice Python, serve come strumento efficace per comprendere cosa fanno i motori autograd e come funzionano.

Iniziamo con un semplice esempio usando la libreria Micrograd originale:

```python
from micrograd.engine import Value

# Crea due oggetti "Value", avvolge due numeri float in essi
a = Value(-4.0)
b = Value(2.0)

# Applica operazioni aritmetiche su questi oggetti "Value"
# Crea un nuovo oggetto "Value" c e lo sovrascrive due volte
c = a + b
c += c + 1
c += 1 + c + (-a)

# Stampa -1.0
print(c.data)

# Applica la Backpropagation
c.backward()

# Stampa 3.0
print(a.grad)
# Stampa 4.0
print(b.grad)
```

Micrograd ti permette di definire valori numerici e applicare operazioni aritmetiche a questi valori.
Micrograd inoltre tiene traccia dell'utilizzo di ogni valore mentre il tempo progredisce in un cosiddetto grafo di espressioni.
Infine, questo grafo di espressioni viene attraversato all'indietro per calcolare i gradienti risultanti dalle operazioni applicate.

> **Il gradiente è un valore che indica la sensibilità del risultato finale ai cambiamenti nei valori che lo influenzano.** Se sappiamo quanto un cambiamento in un valore influenza il risultato finale, possiamo regolare il valore di conseguenza per spostare il risultato finale nella direzione desiderata.
> Questo è il concetto fondamentale dietro la **backpropagation.**

Nel codice sopra, `a` e `b` influenzano `c` attraverso diverse operazioni. Dopo di che, eseguiamo `c.backward()` per calcolare i gradienti per `a` e `b` rispetto a `c`. I gradienti indicano la sensibilità del risultato finale `c` ai cambiamenti nei valori che lo influenzano `a` e `b`.
Per esempio, un piccolo cambiamento nel valore iniziale `b` risulterebbe in un cambiamento 4 volte più grande nel risultato finale `c`.

L'esempio sopra è molto basilare, ma la backpropagation può essere applicata per varie operazioni aritmetiche. Con i Multi-Layered Perceptrons (MLP), una sottoclasse delle reti neurali, è un po' più specifico. Lì abbiamo input e pesi che interagiscono tra loro attraverso moltiplicazione e addizione di matrici.

> Partendo da `c`, il gradiente viene calcolato applicando ricorsivamente la regola della catena a tutti i nodi nel grafo di espressioni che influenzano il valore di `c`.

**Ma cosa significa questo?**

**Cos'è un grafo di espressioni? Perché usiamo i grafi di espressioni? Come si applica qui la regola della catena?**

Analizziamo il problema passo dopo passo.