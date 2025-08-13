# Goal

This course is about building and training neural networks from scratch, starting hands-on from the very basics.
For a start, we will walk through the micrograd project to clearly explain neural network fundamentals.
We will cover the basics of what neural networks actually are. We will look at how they are trained and how they learn through what is called backpropagation.
We will also use step-by-step examples throughout.

**Micrograd is an autograd engine (Automatic Gradient Engine).** It contains the essentials for training neural networks in just **150** lines of code.
Notably, Micrograd implements **backpropagation** from scratch.
To make a neural network learn from data, backpropagation allows to iteratively tune the network's parameters such that they minimize the difference between the network's own prediction and a known reference value. In other words, backpropagation is what makes neural networks learn.

If all this sounds complicated, don't worry about it.
We will approach these concepts one by one, from scratch.
