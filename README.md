# inventing-inductive-priors

Inductive priors (like convolutions, residual connections, batch normalization, attention/transformer, + more) are the drivers behind deep learning. 

We don't see massive vanilla neural networks doing anywhere close to the same performance.

Instead of waiting for a groundbreaking paper with a new inductive prior, let's invent some automatically! Inductive prior search!

**Goals**

- [ ] Verify it might work: simple brute force search example
- [ ] Verify it could really work: search example for something more complicated
- [ ] Verify it works well: rediscover existing inductive priors (Convolutions or attention)
- [ ] Invent: new inductive priors with program search/synthesis

## Experiment 1

### Intro
Let's take a minimal example. You have a straight line of data, say $x=[0,1,2,3]$ $y=[2,4,6,8]$ and we want to find a function to map x to y. 

We know that $f(x) = 2x + 2$ would encode this line perfectly! 

In practice we might learn the weights if the data was more interesting.

For example $f(x) = w \cdot x + b$ where w and b are learned weights that we can optimize via gradient descent and backprop.

Why would we assume that we should multiply the input by w and add b? It's almost like the wx + b itself is an inductive prior. So this becomes the minimal example to show that search for inductive priors would work!  

### Setup

