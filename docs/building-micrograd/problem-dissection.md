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

Calculus taught us that the derivative of $f(x)=3x^2-4x+5$ is $f'(x)=6x - 4$, and $f'(3) = 14$, but we act strictly on the definition here.)

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