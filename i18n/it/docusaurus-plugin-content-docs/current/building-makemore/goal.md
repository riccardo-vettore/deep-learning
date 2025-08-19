# Goal

Questa lezione esaminerà [Makemore](https://github.com/karpathy/makemore).<br/>
Makemore prende un file di testo (come il file `names.txt` fornito). Ogni riga in quel file è considerata un campione di addestramento.<br/>
Makemore quindi impara a generare altri campioni simili a quelli trovati nel file.

Sotto il cofano, Makemore è quello che viene chiamato **character-level language model**.

Ogni singola riga nel file di testo contiene un campione, una sequenza di parole, composta da un insieme di simboli (caratteri, numeri, spazi, punteggiatura, ...).<br/>
Makemore cerca di predire il carattere successivo all'interno di una data sequenza.
> Makemore è un character-level language model implementato in modo moderno.

**Il nostro obiettivo è capire come funziona Makemore. Per farlo, costruiremo Makemore da zero.**

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU if available (faster calculations with PyTorch)
```

```python
words = open('../names.txt', 'r').read().splitlines()   # Python list of strings

print("First 10 names: ", words[:10])   # First ten names, each as separate string
print("Dataset size: ", len(words))     # Amount of words in dataset
print("Shortest name: ", min(len(w) for w in words))    # Smallest word in dataset
print("Longest name: ", max(len(w) for w in words))     # Longest word in dataset
```

First 10 names:  ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia', 'harper', 'evelyn']
Dataset size:  32033
Shortest name:  2
Longest name:  15

Con il character-level approach, ciascuno dei nomi sopra è in realtà una sequenza di esempi di quale carattere segue quale altro,
quale carattere potrebbe venire per primo, quanti caratteri ci sono nelle sequenze di caratteri, ecc.