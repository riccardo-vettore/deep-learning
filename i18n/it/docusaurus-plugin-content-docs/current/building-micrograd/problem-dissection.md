# Problem Dissection

## Comprendere le Derivate

Se vogliamo derivare l'impatto parziale delle variabili contribuenti su altre variabili, calcolare le derivate è essenziale.
**Infatti, le derivate fanno esattamente quello di cui abbiamo bisogno: quantificano l'entità dei cambiamenti negli input che influenzano gli output.**

Implementiamo la funzione quadratica arbitraria $f(x)=3x^2-4x+5$ in Python:

```python
def f(x):
    return 3*x**2 - 4*x + 5 # la nostra funzione di esempio arbitraria

f(3.0) # Stampa 20.0
```

Tracciamo anche la nostra funzione di esempio:
```python
xs = np.arange(-5, 5, 0.25)  # Set di valori da -5 a 5 con passo 0.25
ys = f(xs)  # Applicando f a ogni x
plt.plot(xs, ys);  # Traccia y per ogni x
```
![image](/img/build-micrograd/understanding-derivatives.png)

Ora, qual è la derivata in qualsiasi punto $x$ per la nostra funzione $f(x)$?
Per arriva a risolverlo, dobbiamo prima capire cosa ci sta effettivamente dicendo la derivata su una funzione $f(x)$.

**Questa è la definizione da manuale di cosa significa derivare una funzione:**

$L = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$

**Cosa significa?**
Ci viene chiesto di aggiungere un valore positivo $h$, vicino a $0$, al nostro $a$ per vedere se $f(a + h)$ aumenta o diminuisce il valore restituito dalla funzione, rispetto a $f(a)$.
Se la funzione aumenta, la derivata è positiva. Se la funzione diminuisce, la derivata è negativa.

Implementiamolo:
```python
h = 0.00000001
x = 3.0

# Derivata approssimata di f in x=3
print((f(x+h)-f(x))/h) #14.00000009255109
```

Questo risultato ci dice che la derivata di $f$ rispetto a $x$ in $x = 3$ è $m \approx 14$

(Il calcolo ci ha mostrato che la derivata di $f(x)=3x^2-4x+5$ è $f'(x)=6x - 4$, e $f'(3) = 14$)

Aumentiamo la complessità con $3$ input e $1$ output:

```python
a = 2.0
b = -3.0
c = 10.0
d = a*b + c

print(d) #4.0
```

Dato il codice sopra, `d = a * b + c` è ancora una funzione.
Potresti pensare a prima vista che questa funzione sia 'più semplice' della funzione quadratica di prima,
ma qual è la derivata di `d` rispetto a `a`, `b` e `c`?

Prendiamo di nuovo l'approccio alla lettera:
```python
h = 0.00001

# Questo è il punto (a, b, c) 
# per cui vogliamo la derivata di d
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c # valore della funzione in (a, b, c)

a += h       # aumenta a di h
d2 = a*b + c # valore della funzione in (a+h, b, c)
a -= h       # ripristina a

b += h       # aumenta b di h
d3 = a*b + c # valore della funzione in (a, b+h, c)
b -= h       # ripristina b

c += h       # aumenta c di h
d4 = a*b + c # valore della funzione in (a, b, c+h)

print('Valore della funzione per (a,b,c) d1:\t', d1) #4.0
print()
print('Valore della funzione per (a+h,b,c) d2:', d2) #3.9999699999999994

# Quanto è aumentata la funzione aumentando a
print('pendenza', (d2 - d1)/h) #-3.000000000064062
print('\nValore della funzione per (a,b+h,c) d3:', d3) #4.00002

# Quanto è aumentata la funzione aumentando b
print('pendenza', (d3 - d1)/h) #2.0000000000131024
print('\nValore della funzione di (a,b,c+h) d4:\t', d4) #4.00001
# Quanto è aumentata la funzione aumentando c
print('pendenza', (d4 - d1)/h) #0.9999999999621422
```

Queste funzioni modificate ci dicono come il valore della funzione originale cambia rispetto a ogni singolo valore di input.
**Questa è la derivata parziale della funzione rispetto ad a $(d2)$, b $(d3)$ o c $(d4)$.**

> **La derivata parziale ci dice come l'output di una funzione cambia in relazione a un cambiamento in uno qualsiasi dei suoi input.**

## Derivate nelle Reti Neurali

### Classe Value - Configurazione

Vogliamo spostare la logica delle derivate alle reti neurali.
Per raggiungere questo, abbiamo bisogno di strutture dati adeguate.

La classe `Value` prende un singolo valore numerico e ne tiene traccia.
Puoi definire valori come `a = Value(3.0)` e `b = Value(-2.0)`,
ma dovresti anche essere in grado di eseguire `a + b` o `a * b` per costruire un grafo di operazioni.
E da questo, dovremmo essere in grado di trovare la derivata del risultato finale rispetto ai valori iniziali.
```python
class Value:
    
    # Inizializzazione dell'oggetto
    def __init__(self, data):
        self.data = data

    # Dice come stampare questo oggetto in modo carino  
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addizione, a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
    # Moltiplicazione
    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c # questo è davvero: a.__mul__(b).__add__(c)

print(d) # Value(data=4.0)
```

### Classe Value - Forward

Abbiamo implementato l'archiviazione e presentazione dei dati così come la moltiplicazione e l'addizione.
Quello che ci manca ancora è una struttura per sapere quali operazioni sono state applicate, qual era l'ordine di applicazione e quali connessioni tra oggetti `Value` sono state fatte lungo il percorso.
**In altre parole, vogliamo registrare come specifici Values producono altri Values.**

Per aggiungere questa capacità di tracciamento, estendiamo Value con un attributo `_children`.
`_children` è una tupla vuota che internamente è memorizzata come un `set` (questo cambio concettuale da tupla a set è solo per le prestazioni)

> L'attributo `_children` è un set di oggetti `Value` che influenza direttamente l'oggetto `Value` corrente. Per `c = a + b`, `c` avrebbe `a` e `b` nel suo set `_children`.

Potresti chiederti perché chiamiamo l'attributo `_children` e `_prev` e non `_parents` e `_prev`. È una scelta di design, ma ti assicuro che non è un errore.
Il motivo è che in seguito attraverseremo il grafo all'indietro, dal risultato agli input. Quindi, quello che tecnicamente ora sembrano essere i genitori sembreranno poi i figli.

```python
class Value:
    
    # Questo è stato esteso per prendere _children
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addizione, a+b == a.__add__(b)
    def __add__(self, other):
        # Inizializziamo i _children del risultato come self e other
        out = Value(self.data + other.data, (self, other))
        return out
    
    # Moltiplicazione
    def __mul__(self, other):
        # Inizializziamo i _children del risultato come self e other
        out = Value(self.data * other.data, (self, other))
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c

d # Value(data=4.0)
d._prev # {Value(data=-6.0), Value(data=10.0)}
```

Ora conosciamo i valori immediatamente precedenti, i figli, ma non sappiamo come `d` è stato creato con questi valori.
Per raggiungere questo, estendiamo ulteriormente la nostra classe `Value`
È stato aggiunto anche un attributo label per la generazione del grafo qui sotto. Questo è puramente per scopi di visualizzazione.
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addizione
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # Moltiplicazione
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'

f = Value(-2.0, label='f')
L = d*f; L.label = 'L'

print(L)       # Value(data=-8.0)
print(L._prev) # {Value(data=-2.0), Value(data=4.0)}
print(L._op)   # *
```

## Classe Value - Generazione del Grafo
Ora possiamo tracciare che `d` è stato prodotto dall'addizione di due valori `e` e `c`.
Più in generale, possiamo seguire quale `Value` è stato creato da quale e come è stato fatto.

**Idealmente, vorremmo avere un modo di visualizzare il nostro grafo di espressioni.**
```python
from graphviz import Digraph

# Enumera tutti i nodi e gli archi -> costruisce un set per loro
def trace(root):
    # costruisce un set di tutti i nodi e archi in un grafo
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# Disegna il grafo
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = da sinistra a destra
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # per qualsiasi valore nel grafo, crea un nodo rettangolare ('record') per esso
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
          # se questo valore è il risultato di qualche operazione, crea un nodo op per esso
          dot.node(name = uid + n._op, label = n._op)
          # e collegalo ad esso
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # collega n1 al nodo op di n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

draw_dot(L)
```
![Un'immagine dallo static](/svg/building-micrograd/value-class.svg)

## Riepilogo Rapido
Finora,

- possiamo **costruire espressioni matematiche** con $+$ e $*$,
- possiamo **tenere traccia di quali oggetti Value sono interconnessi attraverso quali operazioni**, risultando in un nuovo `Value`
- possiamo **visualizzare il grafo di espressioni** associato a un `Value` risultante
> Attualmente, visualizziamo solo il passaggio in avanti.

Successivamente, dobbiamo anche coprire la backpropagation.

## Classe Value - Configurazione della Backpropagation
Atteniamoci all'esempio sopra di come `L` è stato creato.
Iniziamo con il risultato del passaggio in avanti (cioè `L`). Poi, al contrario, camminiamo lungo l'albero delle dipendenze calcolando il gradiente per i valori intermedi.
> In sostanza, per nodo, calcoliamo la derivata di `L` rispetto a questo nodo.

La derivata di `L` rispetto a `L` è $1$.
Facile, ma qual è la derivata di `L` rispetto a `f` e così via?
La derivata di un valore (come `L`) rispetto a un altro valore che contribuisce ad esso (come `f`) è chiamata **derivata parziale**, nota anche come **gradiente**.
> Il gradiente rappresenta la derivata della funzione di perdita rispetto al `Value` corrente

Per ogni Value la derivata è impostata a $0$ per default. Questo sarà modificato di conseguenza per ogni interazione matematica nell'albero delle dipendenze a cui partecipa.
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # Questo è come vogliamo stampare questo oggetto
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addizione, a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # Moltiplicazione
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    # Funzione Tanh
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        return Value(t, (self, ), 'tanh')

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'

f = Value(-2.0, label='f')
L = d*f; L.label = 'L'

print(L)       # Value(data=-8.0)
print(L._prev) # {Value(data=-2.0), Value(data=4.0)}
print(L._op)   # *
```
Vediamo come appare nel grafo:

```python
# Enumera tutti i nodi e gli archi -> costruisce un set per loro
def trace(root):
    # costruisce un set di tutti i nodi e archi in un grafo
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# Disegna il grafo
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = da sinistra a destra
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # per qualsiasi valore nel grafo, crea un nodo rettangolare ('record') per esso
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
          # se questo valore è il risultato di qualche operazione, crea un nodo op per esso
          dot.node(name = uid + n._op, label = n._op)
          # e collegalo ad esso
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # collega n1 al nodo op di n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

draw_dot(L)
```
![Un'immagine dallo static](/svg/building-micrograd/setting-up-backpropagation.svg)

In questa struttura del grafo dell'albero delle dipendenze abbiamo un campo per nodo per memorizzare il valore backpropagation.
Ma il calcolo e l'accumulo dei gradienti non è ancora stato fatto.

Partiamo da questa idea:
```python
# Questo è sempre il caso
L.grad = 1.0
```
$L = d * f$

$
\frac{\partial L}{\partial d} = \frac{\partial (d*f)}{\partial d} = \frac{\partial d}{\partial d} * f + d * \frac{\partial f}{\partial d} = 1*f + d*0 = f
$
```python
d.grad = f.data
```
$
\frac{\partial L}{\partial f} = \frac{\partial (d*f)}{\partial f} = \frac{\partial d}{\partial f} * f + d * \frac{\partial f}{\partial f} = 0*f + d*1 = d
$
```python
f.grad = d.data
```
**Arriviamo al sodo della backpropagation.**
>**Se capisci la parte successiva, capirai anche come funziona l'addestramento delle reti neurali a un livello fondamentale.**

Introduciamo una notazione matematica.
Diciamo che dobbiamo determinare la derivata parziale di $L$ rispetto a $c$, o $\frac{\partial L}{\partial c}$.
Diciamo anche che ci atteniamo all'esempio di sopra, cioè:
- $e = a * b$
- $d = e + c$
- $L = d * f$

Per il nostro esempio, assumiamo che questo sia **già noto**: $\frac{\partial L}{\partial d} = f = -2$

Andrò avanti e affermerò che solo da questa impostazione, possiamo dire direttamente che la derivata locale di $c$ e $e$ è $1$, rispettivamente.
Il termine **derivata locale** si riferisce all'operazione immediata all'interno della quale vengono utilizzati i termini $c$ e $e$:
$d = c + e$.
Intuitivamente, un cambiamento di una particolare dimensione a $c$ o $e$ impatta $d$ con esattamente la stessa dimensione.
Pertanto la derivata locale, il 'fattore di impatto' per così dire, è $1$ per entrambi $c$ e $e$.

$\frac{\partial d}{\partial c} = \frac{\partial (e + c)}{\partial c} = \frac{\partial e}{\partial c} + \frac{\partial c}{\partial c} = 0 + 1 = 1 $

$\frac{\partial d}{\partial e} = \frac{\partial (e + c)}{\partial e} = \frac{\partial e}{\partial e} + \frac{\partial c}{\partial e} = 1 + 0 = 1 $

Poiché il nostro obiettivo è trovare $\frac{\partial L}{\partial c}$, dobbiamo in qualche modo concatenare il risultato intermedio noto $\frac{\partial L}{\partial d}$ con la nostra derivata locale $\frac{\partial d}{\partial c}$ per formulare $\frac{\partial L}{\partial c}$ attraverso entrambi.
>**Sembra un lavoro per la regola della catena!**

La regola della catena appare così:

$\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}$

Curiosamente, il $\frac{\partial d}{\partial c}$ risulta non avere effetto sul risultato finale $\frac{\partial L}{\partial c}$ poiché è $1.0$.

>Poiché le derivate locali del nodo di addizione sono $1.0$, l'addizione fondamentalmente instrada il gradiente ugualmente verso i valori che sono stati sommati.
> 
>In altre parole, l'addizione distribuisce uniformemente il gradiente che abbiamo accumulato fino a quel punto agli addendi.

$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} * \frac{\partial d}{\partial c} = -2 * 1 = \underline{\underline{-2.0}} $

$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} * \frac{\partial d}{\partial e} = -2 * 1 = \underline{\underline{-2.0}}$

Andando più avanti nell'albero delle dipendenze, $e$ è composto da $a$ e $b$ attraverso $e = a * b$.
Dobbiamo occuparci della moltiplicazione ora.

Sopra, abbiamo imparato che $\frac{\partial L}{\partial e} = -2.0$.

Ora, di nuovo, iniziamo formulando le derivate locali: $\frac{\partial e}{\partial a}$ e $\frac{\partial e}{\partial b}$

possiamo vedere che:

$\frac{\partial e}{\partial a} = \frac{\partial a * b}{\partial a} = b * \frac{\partial a}{\partial a} + \frac{\partial b}{\partial a} * a = b * 1 + 0 * a = b = -3 $

$\frac{\partial e}{\partial b} = \frac{\partial a * b}{\partial b} = b * \frac{\partial a}{\partial b} + \frac{\partial b}{\partial b} * a = b * 0 + 1 * a = a = 2 $

> La scala con cui uno qualsiasi dei due fattori influenza il risultato è definita dall'altro fattore.

Possiamo usare di nuovo la regola della catena per calcolare $\frac{\partial L}{\partial a}$ e $\frac{\partial L}{\partial b}$

$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} * \frac{\partial e}{\partial a} = -2 * (-3) = 6 $

$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} * \frac{\partial e}{\partial b} = -2 * 2 = -4 $

Combinato con i gradienti per `c` e `e`, questo è il grafo aggiornato:

```python
# Gradienti della somma
c.grad = d.grad * (1) # dL/dc = dL/dd * dd/dc = dL/dd * 1
e.grad = d.grad * (1) # dL/de = dL/dd * dd/de = dL/dd * 1

# Gradienti della moltiplicazione
a.grad = e.grad * b.data # dL/da = dL/de * de/da = dL/de * b
b.grad = e.grad * a.data # dL/db = dL/de * de/db = dL/de * a

draw_dot(L)
```
![Un'immagine dallo static](/svg/building-micrograd/full-backpropagation.svg)

**Perché abbiamo visto un aumento di `L`?**

Il gradiente punta sempre nella direzione della **salita più ripida**.

Muovere un valore in quella direzione fornirà il più grande effetto possibile di aumento sul risultato finale `L` attraverso questo valore.<br/>
Muovere tutti i valori nella direzione del gradiente massimizza `L`.<br/>
Muovere tutti i valori nella direzione esattamente opposta/negativa di questo gradiente quindi minimizza `L`.