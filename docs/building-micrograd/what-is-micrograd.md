# What is Micrograd?

**Autograd engines are the core component enabling neural network training.**
Micrograd is a tiny autograd engine that supports automatic differentiation as well as higher-order gradient calculation.
Consisting of around 150 lines of Python code, it serves as an effective tool for understanding what autograd engines do and how they work.

Let's start with a simple example using the original Micrograd library:

```python live
from micrograd.engine import Value

# Create two "Value" object, wrap two float numbers in them
a = Value(-4.0)
b = Value(2.0)

# Apply arithmetic operations on these "Value" objects
# Create a new "Value" object c and override it twice
c = a + b
c += c + 1
c += 1 + c + (-a)

# Prints -1.0
print(c.data)

# Apply Backpropagation
c.backward()

# Prints 3.0
print(a.grad)
# Prints 4.0
print(b.grad)
```


Micrograd allows you to define numeric values and apply arithmetic operations to these values just as usual.
But, Micrograd additionally keeps track of each value's usages as time progresses in a so-called expression graph.
Finally, this built up expression graph is traversed backwards to compute the gradients resulting from applied operations.

> **The gradient is a value that indicates the sensitivity of the final result to changes in the affecting values.** If we know how much a change in a value affects the final result, we can adjust the value accordingly to shift the final result in the desired direction.
> This is the core concept behind **backpropagation.**

In the above code, `a` and `b` affect `c` through several different operations. After that, we run `c.backward()` to compute the gradients for `a` and `b` with respect to `c`. The gradients indicate the sensitivity of the final result `c` to changes in the affecting values `a` and `b`.
For example, a small change in the initial value `b` would result in a 4x as large a change in the final result `c`.

The above example is very basic, but backpropagation can be applied for various arithmetic operations. With Multi-Layered Perceptrons (MLP), a subclass of neural networks, it's a bit more specific. There we have inputs and weights interacting with each other through matrix multiplication and addition.

> Starting from `c`, the gradient is calculated by recursively applying the chain rule to all nodes in the expression graph that affect the value of `c`.

**But what does this mean?**

**What is an expression graph? Why do we use expression graphs? How does the chain rule apply here?**

Let's dissect the problem step by step.

