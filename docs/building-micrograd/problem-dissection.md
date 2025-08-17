# Problem Dissection

## Understanding Derivatives

If we want to derive the partial impact of contributing variables to some other variables, calculating derivatives is essential.
**In fact, derivatives do exactly what we need here: They quantify the extent of changes in inputs affecting the outputs.**

Let's start slow; let's implement the arbitrary quadratic function $f(x)=3x^2-4x+5$ in Python:

```python
def f(x):
    return 3*x**2 - 4*x + 5 # our arbitrary example function

f(3.0) # Prints 20.0
```

Let's also plot our example function:
```python
xs = np.arange(-5, 5, 0.25)  # Set of values from -5 to 5 with step 0.25
ys = f(xs)  # Applying f to each x
plt.plot(xs, ys);  # Plot y for each x
```
![An image from the static](/img/build-micrograd/understanding-derivatives.png)

Now, what is the derivative at any point $x$ for our function $f(x)$?
To go about solving this, we should first understand what the derivative is actually telling us about a function $f(x)$.

**This is the textbook definition of what it means to derive a function:**

$L = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$

**What does this mean?**
We're asked to add some positive value $h$ which is close to $0$ to our $a$ to see whether this 'nudge $f(a + h)$' increases or decreases the value returned by the function, compared to $f(a)$.
If the function increases, the derivative is positive. If the function decreases, the derivative is negative.

We implement this verbatim:
```python
h = 0.00000001
x = 3.0

# Approximate derivative of f at x=3
print((f(x+h)-f(x))/h) #14.00000009255109
```

This result tells us that the derivative of $f$ with respect to $x$ at $x = 3$ is $m \approx 14$

(Calculus taught us that the derivative of $f(x)=3x^2-4x+5$ is $f'(x)=6x - 4$, and $f'(3) = 14$, but we act strictly on the definition here.)

Let's increase the complexity with $3$ inputs and $1$ output:

```python
a = 2.0
b = -3.0
c = 10.0
d = a*b + c

print(d) #4.0
```

Given the code above, `d = a * b + c` is still a function.
You may think at `a` first glance that this function feels 'simpler' than the quatratic function from before,
but what is the derivative of `d` with respect to `a`,`b` and `c`?

Let's again take the verbatim approach:
```python
h = 0.00001

# This is the point (a, b, c) 
# for which we want the derivative of d
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c # function value at (a, b, c)

a += h       # bump up a by h
d2 = a*b + c # function value at (a+h, b, c)
a -= h       # restore a

b += h       # bump up b by h
d3 = a*b + c # function value at (a, b+h, c)
b -= h       # restore b

c += h       # bump up c by h
d4 = a*b + c # function value at (a, b, c+h)

print('Function value for (a,b,c) d1:\t', d1) #4.0
print()
print('Function value for (a+h,b,c) d2:', d2) #3.9999699999999994

# How much the function increased from bumping up a
print('slope', (d2 - d1)/h) #-3.000000000064062
print('\nFunction value for (a,b+h,c) d3:', d3) #4.00002

# How much the function increased from bumping up b
print('slope', (d3 - d1)/h) #2.0000000000131024
print('\nFunction value of (a,b,c+h) d4:\t', d4) #4.00001
# How much the function increased from bumping up c
print('slope', (d4 - d1)/h) #0.9999999999621422
```

These modified functions tell us how the value of the original function changes with respect to each individual input value.
**This is the partial derivative of the function with respect to either a $(d2)$, b $(d3)$ or c $(d4)$.**

> **The partial derivative tells us how a function output changes in relation to a change in either of its inputs.**

## Derivatives in Neural Networks

### Value Class - Setup

We want to move the logic of derivatives over to neural networks.
To achieve this, we require suitable data structures.

The class `Value` takes a single numeric value and keeps track of it.
You can define values like `a = Value(3.0)` and `b = Value(-2.0)`,
but you then should also be able to perform `a + b` or `a * b` in order to construct a graph of operations.
And from this, we should be able to find the derivative of the final result with respect to the initial values.
```python
class Value:
    
    # Object initialization
    def __init__(self, data):
        self.data = data

    # Tells how to print this object nicely  
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition, a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
    # Multiplication
    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c # this really is: a.__mul__(b).__add__(c)

print(d) # Value(data=4.0)
```

### Value Class - Forward

Data storage and presentation as well as multiplication and addition are accounted for.
What we are still missing now is a structure for knowing which operations were applied, what the order of application was and which connections between `Value` objects were made along the way.
**Put differently, we want to record how specific Values produce other Values.**

To add this tracking capability, we extend Value by an attribute `_children`.
`_children` is an empty tuple that internally is stored as a `set` (this conceptual switch from tuple to set is just for performance)

> The `_children` attribute is a set of `Value` objects that influences the current `Value` object directly. For `c = a + b`, `c` would have `a` and `b` in its `_children` set.

You might ask yourself why we call the attribute `_children` and `_prev` and not `_parents` and `_prev`. It's a design choice, but I assure you it's not a mistake.
The reason is that later we will traverse the graph backwards, from the result to the inputs. So, what technically now seems to be the parents will then seem to be the children.

```python
class Value:
    
    # This got extended to take in _children
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition, a+b == a.__add__(b)
    def __add__(self, other):
        # We initialize the result's _children to be self and other
        out = Value(self.data + other.data, (self, other))
        return out
    
    # Multiplication
    def __mul__(self, other):
        # We initialize the result's _children to be self and other
        out = Value(self.data * other.data, (self, other))
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c

d # Value(data=4.0)
d._prev # {Value(data=-6.0), Value(data=10.0)}
```

We now know the immediate prior values, the children, but we don't know how `d` was created with these values.
To achieve this, we extend our `Value` class further
Also added is a label attribute for the graph generation down below. This is purely for visualization purposes.
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # Multiplication
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

## Value Class - Graph Generation
We can now track that `d` was produced by the addition of two values `e` and `c`.
More generally, we can follow which `Value` was created from which and how that was done, like in a tree making up a final solution node.

**Ideally, we'd like to have a way of visualizing our expression graph.**
```python
from graphviz import Digraph

# Enumerates all the nodes and edges -> builds a set for them
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# Draw the graph
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
          # if this value is a result of some operation, create an op node for it
          dot.node(name = uid + n._op, label = n._op)
          # and connect this node to it
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

draw_dot(L)
```
![An image from the static](/svg/building-micrograd/value-class.svg)

## Quick Recap
So far,

- we can **build mathematical expressions** with $+$ and $*$,
- we can **keep track of what Value objects are interconnected through what operations**, resulting in a new `Value`
- we can **visualize the expression graph** associated to a resulting `Value`
> Currently, we only visualize the forward pass.

Next, we need to also cover backpropagation.

## Value Class - Setting up Backpropagation
Let's stick with the above example of how `L` was created.
We start with the forward pass result (meaning `L`). Then, in reverse, we walk along the dependency tree calculating the gradient for the intermediate values.
> In essence, per node, we calculate the derivative of `L` with respect to this node.

The derivative of `L` with respect to `L` is $1$.
Easy, but what's the derivative of `L` with respect to `f` and so on?
The derivative of one value (like `L`) with respect to another value that contributes to it (like `f`) is called the **partial derivative**, also known as the **gradient**.
> The **gradient** is the derivative of the loss function with respect to the current `Value`.

For every Value the derivative is set to $0$ by default. This will be modified accordingly for every mathematical interaction in the dependency tree that it participates in.
```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # This is how we want to print this object
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Addition, a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # Multiplication
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    # Tanh function
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
Let's see what this looks like in the graph:

```python
# Enumerates all the nodes and edges -> builds a set for them
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# Draw the graph
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
          # if this value is a result of some operation, create an op node for it
          dot.node(name = uid + n._op, label = n._op)
          # and connect this node to it
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

draw_dot(L)
```
![An image from the static](/svg/building-micrograd/setting-up-backpropagation.svg)

In this dependency tree graph structure we have a field per node to store the backpropagated value.
But calculation and accumulation of the gradients is still missing.

Let's play with this idea first:
```python
# This is just always the case
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
**Let's get to the crux of backpropagation.**
>**If you get the next part, you will also get how training neural networks works on a fundamental level.**

Let's introduce some mathematical notation.
Say, we need to determine the partial derivative of $L$ with respect to $c$, or $\frac{\partial L}{\partial c}$.
Let's also say that we stick to the example from above, meaning:
- $e = a * b$
- $d = e + c$
- $L = d * f$

For our example, we assume this is **known already**: $\frac{\partial L}{\partial d} = f = -2$

I'll go ahead and claim that just from this setting, we can directly say that the local derivative of $c$ and $e$ is $1$, respectively.
The term **local derivative** refers to the immediate operation within which the terms $c$ and $e$ are used:
$d = c + e$.
Intuitively, a change of a particular size to $c$ or $e$ impacts $d$ with the exact same size.
Therefore the local derivative, the 'impact factor' so to say, is $1$ for both $c$ and $e$.

$\frac{\partial d}{\partial c} = \frac{\partial (e + c)}{\partial c} = \frac{\partial e}{\partial c} + \frac{\partial c}{\partial c} = 0 + 1 = 1 $

$\frac{\partial d}{\partial e} = \frac{\partial (e + c)}{\partial e} = \frac{\partial e}{\partial e} + \frac{\partial c}{\partial e} = 1 + 0 = 1 $

As our goal is to find $\frac{\partial L}{\partial c}$, we need to somehow concatenate the known intermediary result $\frac{\partial L}{\partial d}$ with our local derivative $\frac{\partial d}{\partial c}$ to formulate $\frac{\partial L}{\partial c}$ across both of them.
>**That looks like a job for the chain rule!**

The chain rule looks like this:

$\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}$

Curiously, the $\frac{\partial d}{\partial c}$ turns out to have no effect on the final result $\frac{\partial L}{\partial c}$ as it's $1.0$.

>As the addition node's local derivatives are $1.0$, addition basically routes the gradient equally down to the values that were summed.
> 
>In other words, addition evenly distributes the gradient we accumulated until that point to the summands.

$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} * \frac{\partial d}{\partial c} = -2 * 1 = \underline{\underline{-2.0}} $

$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} * \frac{\partial d}{\partial e} = -2 * 1 = \underline{\underline{-2.0}}$

Stepping further into the dependency tree, $e$ is made up of $a$ and $b$ through $e = a * b$.
We have to deal with multiplication now.

Above, we learnt that $\frac{\partial L}{\partial e} = -2.0$.

Now, again, we start by formulating the local derivatives: $\frac{\partial e}{\partial a}$ and $\frac{\partial e}{\partial b}$

we can see that:

$\frac{\partial e}{\partial a} = \frac{\partial a * b}{\partial a} = b * \frac{\partial a}{\partial a} + \frac{\partial b}{\partial a} * a = b * 1 + 0 * a = b = -3 $

$\frac{\partial e}{\partial b} = \frac{\partial a * b}{\partial b} = b * \frac{\partial a}{\partial b} + \frac{\partial b}{\partial b} * a = b * 0 + 1 * a = a = 2 $

> The scale by which either of two factors affects the result is defined by the other factor.

We can use the chain rule again to calculate $\frac{\partial L}{\partial a}$ and $\frac{\partial L}{\partial b}$

$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} * \frac{\partial e}{\partial a} = -2 * (-3) = 6 $

$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} * \frac{\partial e}{\partial b} = -2 * 2 = -4 $

Combined with the gradients for `c` and `e`, this is the updated graph:

```python
# Gradients of sum
c.grad = d.grad * (1) # dL/dc = dL/dd * dd/dc = dL/dd * 1
e.grad = d.grad * (1) # dL/de = dL/dd * dd/de = dL/dd * 1

# Gradients of multiplication
a.grad = e.grad * b.data # dL/da = dL/de * de/da = dL/de * b
b.grad = e.grad * a.data # dL/db = dL/de * de/db = dL/de * a

draw_dot(L)
```
![An image from the static](/svg/building-micrograd/full-backpropagation.svg)

**Why did we see an increase of `L`?**

The gradient always points in the direction of the **steepest ascent**.

Moving a value in that direction will provide a biggest possible effect of increase on the final result `L` through this value.<br/>
Moving all values in gradient direction maximizes `L`.<br/>
Moving all values in the exact opposite/negative direction of this gradient therefore minimizes `L`.





