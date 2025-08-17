# Neural Networks

Let's take the concepts of forward pass and backpropagation and apply them to neural networks.

Eventually, we want to build working neural networks (NNs):

![An image from the static](/img/neural-networks/img.png)

We can utilize the mathematical model for what a neuron in such a neural network looks like:

![An image from the static](/img/neural-networks/img_1.png)

The activation function used here is `tanh`.
For reference, here's what `tanh` and `sigmoid` (another activation function) look like:

```python
def sigmoid(x):
    a = []
    for i in x:
        a.append(1/(1+math.exp(-i)))
    return a

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Tanh vs. Sigmoid')

lower = -5
upper = 5
step = 0.2

ax1.grid()
ax1.set_title('Tanh')
ax1.plot(np.arange(lower, upper, step), np.tanh(np.arange(-5, 5, 0.2))) # Tanh activation function
ax2.grid()
ax2.set_title('Sigmoid')
ax2.plot(np.arange(lower, upper, step), sigmoid(np.arange(-5, 5, 0.2))) # Sigmoid activation function (alternative)

plt.show();
```
![An image from the static](/img/neural-networks/img_2.png)

The tanh function's range is $[-1,1]$, while the `sigmoid` function's range is $[0,1]$.<br/>
Per iteration, for our specific case, `tanh` will produce rather more precise gradients, leading to more efficient training cycles.

**Don't generalize this notion, though.**

Let's go ahead and build a tiny MLP just by implementing $\text{tanh}(\sum_{i=1}^{n} w_ix_i + b)$ for $n=2$:
```python
# Inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias b
# Making sure backprop numbers come out nice later on
b = Value(6.8813735870195432, label='b')

# Neuron value n: x1w1+x2w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'

# Squashed activation o: tanh(n)
o = n.tanh(); o.label='o'

draw_dot(o)
```
![An image from the static](/svg/neural-networks/svg.svg)

## Manual Propagation
We laid out the ground rules to backpropagation before. But now, let's perform actual manual backpropagation for this setup: Starting from `o` and going backwards, we want to find all the gradients
> By the end, we can ideally answer "What is the derivative of `o` with respect to `x1, w1, x2, w2`?"<br/>
> These are the variables we can directly change and therefore care about the most.

The gradient of the activated result `o` is $1.0$, as always. But what about the gradient of `n`, i.e. $\frac{\partial o}{\partial n}$?

The local derivative of `tanh` is $1 - \text{tanh}^2(n)$<br/>
With that defined, we can go back(wards) to the future:
```python
# Always a given
o.grad = 1.0

# o = tanh(n), what is do/dn?
# do/dn = 1 - tanh(n)**2
n.grad = 1 - o.data**2

# As addition "just" splits the gradient
x1w1x2w2.grad = n.grad
b.grad = n.grad

# And addition again
x1w1.grad = x1w1x2w2.grad
x2w2.grad = x1w1x2w2.grad

# And multiplication handles like in the example above
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data # This will be 0: Changing this value does nothing, as its multiplied by 0
```

- $\frac{\partial o}{\partial o} = 1$

- $\frac{\partial o}{\partial n} = \frac{\partial tanh(n)}{\partial n} = 1 - \text{tanh}^2(n) = 1 - (0.7)^2 = 0.51$

We have $n = x1w1 + x2w2 + b$

$\frac{\partial n}{\partial (x1w1 + x2w2)} = \frac{\partial (x1w1 + x2w2 + b)}{\partial (x1w1 + x2w2)} = \frac{\partial  (x1w1 + x2w2)}{\partial (x1w1 + x2w2)} + \frac{\partial b}{\partial (x1w1 + x2w2)} = 1 + 0 = 1$

$\frac{\partial n}{\partial b} = \frac{\partial (x1w1 + x2w2 + b)}{\partial b} = \frac{\partial (x1w1 + x2w2)}{\partial b} + \frac{\partial b}{\partial b} = 0 + 1 = 1$

We can use the chain rule to calculate $\frac{\partial o}{\partial x1w1 + x2w2}$

- $\frac{\partial o}{\partial (x1w1 + x2w2)} = \frac{\partial o}{\partial n} * \frac{\partial n}{\partial (x1w1 + x2w2)} = 0.51 * 1 = 0.51$

We can use the chain rule again to calculate $\frac{\partial o}{\partial b}$

- $\frac{\partial o}{\partial b} = \frac{\partial o}{\partial n} * \frac{\partial n}{\partial b} = 0.51 * 1 = 0.51$

$\frac{\partial x1w1 + x2w2}{\partial x1w1} = 1$

$\frac{\partial x1w1 + x2w2}{\partial x2w2} = 1$

We can use the chain rule to calculate $\frac{\partial o}{\partial x1w1}$

- $\frac{\partial o}{\partial x1w1} = \frac{\partial o}{\partial x1w1 + x2w2} * \frac{\partial x1w1 + x2w2}{\partial x1w1} = 0.51 * 1 = 0.51$

- $\frac{\partial o}{\partial x2w2} = \frac{\partial o}{\partial x1w1 + x2w2} * \frac{\partial x1w1 + x2w2}{\partial x2w2} = 0.51 * 1 = 0.51$

$\frac{\partial x1w1}{\partial x1} = \frac{\partial x1}{\partial x1} * w1 + x1 * \frac{\partial w1}{\partial x1} = w1 = -3$

$\frac{\partial x1w1}{\partial w1} = \frac{\partial x1}{\partial w1} * w1 + x1 * \frac{\partial w1}{\partial w1} = w1 = 2$

We can use the chain rule to calculate $\frac{\partial o}{\partial x1}$

- $\frac{\partial o}{\partial x1} = \frac{\partial o}{\partial x1w1} * \frac{\partial x1w1}{\partial x1} = 0.51 * (-3) = -1.53$

- $\frac{\partial o}{\partial w1} = \frac{\partial o}{\partial x1w1} * \frac{\partial x1w1}{\partial w1} = 0.51 * 2 = 1.02$

$\frac{\partial x2w2}{\partial x2} = \frac{\partial x2}{\partial x2} * w2 + x2 * \frac{\partial w2}{\partial x2} = w2 = 1$

$\frac{\partial x2w2}{\partial w2} = \frac{\partial x2}{\partial w2} * w2 + x2 * \frac{\partial w2}{\partial w2} = x2 = 0$

We can use the chain rule to calculate $\frac{\partial o}{\partial x2}$

- $\frac{\partial o}{\partial x2} = \frac{\partial o}{\partial x2w2} * \frac{\partial x2w2}{\partial x2} = 0.51 * 1 = 0.51$

- $\frac{\partial o}{\partial w2} = \frac{\partial o}{\partial x2w2} * \frac{\partial x2w2}{\partial w2} = 0.51 * 0 = 0$

## Automated Backpropagation

**Doing backprop manually is amateur-hour**. We should automate and generalize this.

To do so, we need to re-write/extend the `Value` class.
To be more precise, we need to extend the `Value` object by a `_backward` attribute, a lambda expression initialized with `None`.
For each operation, `_backward` is populated with a concrete gradient calculation step.

The code below realizes the above concept really beautifully:
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # Does nothing by default
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition, a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Route gradient to parents
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out
     
    # Multiplication
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Route gradient affected by data of other node
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        
        out._backward = _backward
        return out
    
    # Tanh activation function
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            # Local derivative times gradient of child node
            self.grad = (1 - t**2) * out.grad
        
        out._backward = _backward
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
Let's now build a simple MLP and 'forward pass' through it:

```python
# Inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias
b = Value(6.8813735870195432, label='b') # Making sure backprop numbers come out nice later on

# Forward Pass 
# Goal: Find the value of the single output neuron 'n'
# Neuron value n = x1*w1+x2*w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'

# Squashed activation: tanh(n)
o = n.tanh(); o.label='o'

draw_dot(o)
```
![An image from the static](/svg/neural-networks/svg_1.svg)
We don't have to perform backpropagation manually anymore (except for `o.grad`, please ignore that and keep the exact order of operations in mind):

```python
o.grad = 1.0  # Base case for backprop multiplication to work
o._backward()
n._backward()
b._backward() # Nothing happens, as this is a leaf
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()

draw_dot(o)
```
![An image from the static](/svg/neural-networks/svg_2.svg)

**We now have one last thing to get rid of**: Having to call `_backward()` manually in the particular order (one after another, backwards) on our nodes.

Going backwards through our expression graph means that for every node everything following it has to be already calculated.<br/>
This requires ordering. Ordering graphs can be done using **topological sort**.<br/>
This arranges the nodes of our dependency graph so that edges always point in one common direction:

![An image from the static](/img/neural-networks/img_3.png)

The ordering from topological sort guarantees that gradients are computed and propagated in a manner that avoids redundant computations and maximizes computational parallelism, resulting in improved efficiency through seamless flow of gradients through the network and thus accelerated convergence rates.

```python
# Topological sort
# Structuring our graph so that we can traverse in dependency-respecting order

topo = []
visited = set()

def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v) # Only add node if all preceeding nodes were processed first
build_topo(o)

for t in topo:
    print(t)
```

Value(data=0.0)<br/>
Value(data=1.0)<br/>
Value(data=0.0)<br/>
Value(data=-3.0)<br/>
Value(data=2.0)<br/>
Value(data=-6.0)<br/>
Value(data=-6.0)<br/>
Value(data=6.881373587019543)<br/>
Value(data=0.8813735870195432)<br/>
Value(data=0.7071067811865476)

This is the exact order by which we need to apply `_backward()` to our graph, starting from `o`.

```python
o.grad = 1.0

topo = []
visited = set()

def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v) # Only add node if all nodes were processed first
build_topo(o)

for node in reversed(topo):
    node._backward()

draw_dot(o)
```
We now go on and implant this logic into our Value class.<br/>
**This is the new class structure**:
![An image from the static](/svg/neural-networks/svg_3.svg)

## Value Class - Bug Hunting and Extending

We built backpropagation. At least for one neuron `o`.<br/>
But we still have a bug.

The bug emerges here:

```python
a = Value(3.0, label='a')
b = a + a; b.label = 'b'
b.backward()
draw_dot(b)
```
![An image from the static](/svg/neural-networks/svg_4.svg)

```
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b; d.label='d'
e = a + b; e.label='e'
f = d * e; f.label='f'

f.backward()
draw_dot(f)
```
![An image from the static](/svg/neural-networks/svg_5.svg)

**We have a problem as soon as a variable is used more than once**.<br/>
But this is the case most of the time in real-world examples.

> **We actually need to accumulate the gradients (`+=`) rather than setting/overriding them (`=`)**.

Also, while we're at it, let's extend the `Value` class further.<br/>
For example, we can't do `a = Value(2.0) + 1.0` or `a = Value(2.0) * 2.0`.
And also, let's access division and the detail operations behind `tanh` now.

A fixed and extended version of the `Value` class looks like this (see the backward-functions for the `+=` bugfix):

```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # Does nothing by default
        self._prev = set(_children)
        self._op = _op
        self.label = label
            
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Extension
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad  # Bugfix
            other.grad += 1.0 * out.grad # Bugfix
        
        out._backward = _backward
        return out
    
    # Negation (special multiplication)
    def __neg__(self): # -self
        return -1 * self
    
    # Subtraction (special addition)
    def __sub__(self, other): # self - other
        return self + (-other)
 
    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Extension
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += out.grad * other.data # Bugfix
            other.grad += out.grad * self.data # Bugfix
        
        out._backward = _backward
        return out
    
    # Power (special multiplication)
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers (for now)"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    # Called if self is on right side of *
    def __rmul__(self, other): # other * self
        return self * other
    
    # Called if self is on right side of +
    def __radd__(self, other): # other + self
        return self + other
    
    # True division (special multiplication)
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    # Tanh activation function
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad # Bugfix
        
        out._backward = _backward
        return out
    
    # Exponential function
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # Only add node if all nodes were processed first
        build_topo(self)
        
        self.grad = 1.0 # Seed gradient always 1.0
        
        for node in reversed(topo):
            node._backward()
```

```python
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b; d.label='d'
e = a + b; e.label='e'
f = d * e; f.label='f'

f.backward()
draw_dot(f)
```

![An image from the static](/svg/neural-networks/svg_6.svg)

## Everything comes together

We will now change the example from above.
More precisely, we will change how we define `o` using the now available, more detailed forward and backprop operations for `tanh`.

```python
# Inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias b
b = Value(6.8813735870195432, label='b') # Making sure backprop numbers come out nice later on

# Neuron value n: x1w1+x2w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'

# Squashed activation o: tanh(n) NOW EXPLICITLY IMPLEMENTED
e = (2*n).exp()
o = (e - 1)/(e + 1); o.label='o'

draw_dot(o)
```

![An image from the static](/svg/neural-networks/svg_7.svg)

```python
# The x1, x2,... gradients should remain the same
o.backward()
draw_dot(o)
```

![An image from the static](/svg/neural-networks/svg_8.svg)

**Why did we do what we just did?**

We essentially went ahead and changed the level of implementation in this run.<br/>
It is up to us if we want to just implement one tanh or if we go on and implement that function's atomic steps explicitly.<br/>
This in essence was training on how to deal with operations, with zooming in and out of an implementation approach and with new gradient calculations necessary for the necessary steps.

## Doing the exact same thing using the PyTorch API
The original micrograd is modeled roughly after the PyTorch syntax. In fact, it can just as well be implemented in PyTorch.
This might seem a bit messy at first though, as it requires values to be stored in PyTorch's Tensor objects.

```python
x1 = torch.Tensor([2.0]).double();  x1.requires_grad = True  # single element tensors
x2 = torch.Tensor([0.0]).double();  x2.requires_grad = True  # tensor datatype is now double
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True  # default dtype was float32
w2 = torch.Tensor([1.0]).double();  w2.requires_grad = True  # now its float64 aka double
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b # perform arithmetic just like with micrograd
o = torch.tanh(n)

print(o.data.item()) #0.7071066904050358
o.backward() # backward() is pytorch's autograd function

print('---') # These values below are just like micrograds left most layer
print('x2', x2.grad.item()) # x2 0.5000001283844369
print('w2', w2.grad.item()) # w2 0.0
print('x1', x1.grad.item()) # x1 -1.5000003851533106
print('w1', w1.grad.item()) # w1 1.0000002567688737
```

```python
o.item() # Pluck out the scalar value from tensor o
# 0.7071066904050358
```

> The big deal about being able to use PyTorch is that it makes processing **significantly more efficient** through a bunch of optimizations under the hood.

## Back to Neural Networks

Now that we have some vehicles to build complex mathematic expressions, we can build layered NNs.<br/>
We'll do that piece by piece and end up with a 2-layer multi-layer perceptron (MLP).

In theory, we can say that the output of micrograd's forward pass can be interpreted as the activation of a neuron.<br/>
Therefore we can build a neuron taking in and processing some input. But we will now do so utilizing the powers of PyTorch.

For the sake of completeness, here is the schematic for an NN again:

![An image from the static](/img/neural-networks/img.png)

```python
# One neuron is able to take multiple inputs and produce one activation scalar
class Neuron:
    def __init__(self, nin):
        # nin -> number of inputs to this neuron
        # Random weight [-1, 1] per input
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        # Bias controls general "trigger happiness" of neuron
        self.b = Value(np.random.uniform(-1,1))
        
    def __call__(self, x): # running neuron(x) -> __call__ triggered
        # w * x + b
        # zip() creates iterator running over the tupels of two iterators
        # self.b is taken as the sum's start value and then added upon
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # Squash the activation with tanh
        out = act.tanh()
        return out
    
    # Convenience code to gather the neuron's parameter list
    def parameters(self):
        return self.w + [self.b]


# A set of neurons making up a (hidden/input/output) NN layer
# E.g. n = Layer(2, 3) -> 3 2-dimensional neurons
class Layer:
    # nout -> how many neurons/outputs should be in this layer
    # nin -> how many inputs are to be expected per neuron
    def __init__(self, nin, nout):
        # literally create a list of neurons as needed
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x): # running layer(x) -> __call__ triggered
        # return all of the layer's neuron activations
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # Convenience code to gather all parameters of layer's neurons
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


# MLP -> Multi-layer perceptron -> NN
class MLP:
    # nin -> number of inputs to the NN
    # nouts -> list of numbers, defines sizes of all wanted layers
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x): # mlp(x) -> call all layer(x)s values in NN
        for layer in self.layers:
            # Neat forward pass implementation
            x = layer(x)
        return x
    
    # Convenience code to gather all parameters of all layer's neurons
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Python code to JavaScript
```javascript
// One neuron is able to take multiple inputs and produce one activation scalar
class Neuron {
    constructor(nin) {
        // nin -> number of inputs to this neuron
        // Random weight [-1, 1] per input
        this.w = Array.from({length: nin}, () => new Value(Math.random() * 2 - 1));
        // Bias controls general "trigger happiness" of neuron
        this.b = new Value(Math.random() * 2 - 1);
    }
    
    // Running neuron(x) -> call() triggered
    call(x) {
        // w * x + b
        // Calculate weighted sum of inputs plus bias
        let act = this.b;
        for (let i = 0; i < this.w.length; i++) {
            act = act.add(this.w[i].mul(x[i]));
        }
        // Squash the activation with tanh
        const out = act.tanh();
        return out;
    }
    
    // Convenience method to gather the neuron's parameter list
    parameters() {
        return [...this.w, this.b];
    }
}

// A set of neurons making up a (hidden/input/output) NN layer
// E.g. new Layer(2, 3) -> 3 2-dimensional neurons
class Layer {
    // nout -> how many neurons/outputs should be in this layer
    // nin -> how many inputs are to be expected per neuron
    constructor(nin, nout) {
        // literally create a list of neurons as needed
        this.neurons = Array.from({length: nout}, () => new Neuron(nin));
    }
    
    // Running layer(x) -> call() triggered
    call(x) {
        // return all of the layer's neuron activations
        const outs = this.neurons.map(n => n.call(x));
        return outs.length === 1 ? outs[0] : outs;
    }
    
    // Convenience method to gather all parameters of layer's neurons
    parameters() {
        return this.neurons.flatMap(neuron => neuron.parameters());
    }
}

// MLP -> Multi-layer perceptron -> NN
class MLP {
    // nin -> number of inputs to the NN
    // nouts -> array of numbers, defines sizes of all wanted layers
    constructor(nin, nouts) {
        const sz = [nin, ...nouts];
        this.layers = [];
        for (let i = 0; i < nouts.length; i++) {
            this.layers.push(new Layer(sz[i], sz[i + 1]));
        }
    }
    
    // mlp(x) -> call all layer(x) values in NN
    call(x) {
        let result = x;
        for (const layer of this.layers) {
            // Neat forward pass implementation
            result = layer.call(result);
        }
        return result;
    }
    
    // Convenience method to gather all parameters of all layer's neurons
    parameters() {
        return this.layers.flatMap(layer => layer.parameters());
    }
}
```

```python
x = [2.0, 3.0, -1.0]  # input values
n = MLP(3, [4, 4, 1]) # 3 inputs into 2 layers of 4 and one output layer
print(n(x)) # Value(data=0.4484457497844679)
```

With micrograd, we are now able to relatively easily backpropagate through this mess.<br/>
**Let's do that, let's define an example training set with features and labels.**

```python
# Features/Inputs
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# Desired targets
ys = [1.0, -1.0, -1.0, 1.0]

# Get the NN's current prediction for xs
ypred = [n(x) for x in xs]

for i in range(len(ypred)):
    print(f'{ypred[i]}\t --> {ys[i]}')
    
# Value(data=0.4484457497844679)	 --> 1.0
# Value(data=0.6835797869668085)	 --> -1.0
# Value(data=-0.4890066181780964)	 --> -1.0
# Value(data=0.9233721935145033)	 --> 1.0
```

**The neural network doesn't perform very well at this point. That's because it is untrained.**<br/>
We need to measure how good/bad the neural network performs to make steps towards improving its performance.

**We need a loss function** to measure just how good/bad predictions are.
```python
# Squaring does two things:
# - Makes all errors positive (so they don't cancel out)
# - Penalizes larger errors more heavily than smaller ones
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print('loss:', loss.data) # 3.405639047006587
```

This loss now has to become as low as possible.<br/>
If loss is low, the predictions more closely resemble the expectations provided through the labels.

```python
loss.backward()
```
```python
# Example neuron weight with now calculated gradient
print(n.layers[0].neurons[0].w[0].grad) # First layer's first neuron's first weight's gradient
print(n.layers[0].neurons[0].w[0].data) # First layer's first neuron's first weight's value

# 3.1657504853266514
# -0.0012205999844798754
```
```python
# Weight Update with Backpropagation's gradients
for p in n.parameters(): 
    p.data += -0.01 * p.grad # Move a tiny bit in opposite direction of gradient to not overfit this single example
```
```python
# Show updated weight
print(n.layers[0].neurons[0].w[0].data)

# -0.03287810483774639
```
Ok. We have a loss function as a way to measure how well the neural network performs.
The `loss` term is directly linked to the parameters, the activated weights and biases of the MLP.
If we calculate the gradients, we determine the strength and direction
of the changes we would need to make to the weights and biases to maximize the loss.

Yes, maximize.

That's exactly what we don't want here, though.
We want to minimize the loss. To do so, wenegate the gradients and scale them by a factor of $0.01$.
Changing all parameters by nudging them in the opposite direction of the gradients
by a small amount and only a small amount avoids over-adapting to what individual inputs desire the weights to be.
This process is called **gradient descent** and it is used for training neural networks.

Congratulations, you just learned how to build and train a neural network from scratch!

One more thing.

Training most often is not done by exposing the network once to the data points,
but by doing so multiple times. One such iteration through the dataset during training is called an **epoch**.

We will now train the network for `5` epochs:
```python
# Run epochs and show respective predictions
for t in range(5):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    print(f'Epoch {t}\t - Loss: {loss.data}\t - Predictions: {[y.data for y in ypred]}')

    loss.backward()
    for p in n.parameters(): 
        p.data += -0.01 * p.grad
```

```
Epoch 0	- Loss: 2.794865490485899 - Predictions: [0.389465006232113, 0.49168719307518943, -0.5655991426041365, 0.9090179963951852]
Epoch 1	- Loss: 1.4281126882440753 - Predictions: [0.24237523318167262, -0.1380604692430543, -0.6938845517661758, 0.8678221704723083]
Epoch 2	- Loss: 0.9230393194235128 - Predictions: [0.14188145711589098, -0.6724349178987212, -0.7935867083421373, 0.808253928861854]
Epoch 3	- Loss: 0.7171814437886361 - Predictions: [0.2167705829772557, -0.8279799900516586, -0.8364295667271546, 0.7823145534603788]
Epoch 4	- Loss: 0.4490150472482478 - Predictions: [0.3889621976730381, -0.8773646275220024, -0.8480091165043534, 0.8063322739146247]
```

Desired targets [1.0, -1.0, -1.0, 1.0]