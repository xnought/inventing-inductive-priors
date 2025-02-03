# Can we invent inductive priors from scratch?

WORK IN PROGRESS. Let me know if you want to contribute by contacting me.

Inductive priors (like convolutions, residual connections, batch normalization, attention/transformer, + more) are the drivers behind deep learning. 

We don't see massive vanilla neural networks doing anywhere close to the same performance.

Instead of waiting for a groundbreaking paper with a new inductive prior, let's invent some automatically! Inductive prior search!

**Goals**

- [x] See experiment 1. Verify it might work: simple brute force search example
- [ ] Extends experiment 1 to multiple possible linear like datasets
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

Weight shapes are fixed: We have one weight w and one weight b. These are initialized to 0 so we have default no contribution.

x is the input data (aka simple line I mentioned before).

The model template will have two lines and only one operation per line:

```python
def forward(x, w, b):
	op1 = ??
	op2 = ??
	return op2
```

These ?? are what we can replace with an operation. For example, if we learned the best prior we should get

```python
# f(x) = wx + b
def forward(x, w, b):
	op1 = x@w
	op2 = op1+b
	return op2
```

So I will have a few operations we can search the template over: 
- `@` matmul/dot 
- `*` elementwise multiply
- `+` elementwise add
- `-` elementwise subtract

we know that `-` and `*` need not be used, they are there to show that they are not part of the optimal answer in our search.

If the given template throws an error (for example matmul with wrong dimensions) then the entire program is invalid and we just label it as such.

If the given template returns a shape that doesn't correspond to the expected output (in this case `[4, 1]`), this is also considered an invalid program and we just label it as such.

So with the given setup, I can calculate the total number of possible programs:

4 operations, 2 weights, 2 lines of code (1 op per line). And in the second line (op2 line) may or may not reference the first line (op1).

Note I know plus and multiply are associative, which would reduce number of programs, but ignore for now.


First compute number of op1 combinations: it has access to x, w, and b.

So we can do x+w, w+x, x+b, b+x, ... So if I only had + it would be 3 permute 2 = 6 possible combinations. We can do this for every binary op we have and we have 4 in total, so there are 24 possible op1 combinations. 

Then for op2, we also have access to op1. So we do do the same but 4 permute 2 = 12 and for the 4 possible ops are then 48 possible combinations of op2.

Then since each op1 can have an unique op2, 24 times 48 is 1152 possible programs.

Note that alot of these programs are invalid or duplicates, but for more complicated examples we might not know that for sure. So proceed.

Now for each of the 1152 possible programs I will train the model for 50 iterations using the entire dataset with a learning rate of 0.1 with normal SGD and MSE loss.

If a program errors, it gets thrown in the garbage, if it succeeds and optimizes, save the program / inductive prior and label it with the smallest loss it could achieve.

### Results

See [experiment1.ipynb](./experiment1.ipynb) for results. 

In summary, I get the following inductive priors where each line represents `(lowest_loss_achieved, (op1, op2))`:

```python
[(0.0, ('x@w', 'w+op1')),
 (0.0, ('x@w', 'op1+w')),
 (0.0, ('x@b', 'b+op1')),
 (0.0, ('x@b', 'op1+b')),
 (0.0, ('x*w', 'w+op1')),
 (0.0, ('x*w', 'op1+w')),
 (0.0, ('x*b', 'b+op1')),
 (0.0, ('x*b', 'op1+b')),
 (0.0, ('w*x', 'w+op1')),
 (0.0, ('w*x', 'op1+w')),
 (0.0, ('b*x', 'b+op1')),
 (0.0, ('b*x', 'op1+b')),
 (1.2768488488745788e-09, ('x+w', 'x+op1')),
 (1.2768488488745788e-09, ('x+w', 'op1+x')),
 (1.2768488488745788e-09, ('x+b', 'x+op1')),
 (1.2768488488745788e-09, ('x+b', 'op1+x')),
 (1.2768488488745788e-09, ('w+x', 'x+op1')),
 (1.2768488488745788e-09, ('w+x', 'op1+x')),
 (1.2768488488745788e-09, ('b+x', 'x+op1')),
 ...
 ...
 (53.5, ('x+b', 'b-op1')),
 (53.5, ('w+x', 'w-op1')),
 (53.5, ('b+x', 'b-op1')),
 (53.5, ('w*b', 'op1-x')),
 (53.5, ('b*w', 'op1-x')),
 (53.5, ('w-x', 'op1-w')),
 (53.5, ('b-x', 'op1-b'))]
 ```

 See [experiment1_valid_programs.txt](./experiment1_valid_programs.txt) for all of them.

Interesting enough I get many optimal programs which I did not expect (even though I should have).

I mentioned in the setup that w=2 and b=2 for w*x + b works. So my search didn't care which was the slope or bias and sometimes reused the same parameter for both.

`[(0.0, ('x@w', 'w+op1'))`

Which represents

```python
def forward(x,w,b):
	op1 = x@w
	op2 = w+op1
	return op2
```
reuses the w parameter, because w can repeat for this particular dataset!

Using multiple datasets in the search during training and averaging the best loss would have removed these types of clever parameter reuse programs.

And just because I'm curious:

Looking at the worst program `(53.5, ('b-x', 'op1-b'))`

Which represents

```python
def forward(x,w,b):
	op1 = b-x
	op2 = op1-b
	return op2
```

Is especially bad because it corresponds to b-x-b which is just x.


Main result finding:

**Program search can be used to search for inductive priors for the minimal one line regression case. We find many versions of the y=mx+b structure. Even some programs reuse the same weight as y=mx+m since the particular dataset allowed it (y=2x+2 generated the simple data). In the future you can remove those clever uses with multiple and diverse datasets (multiple linear regression datasets in this case).**

Next steps:

Move to the general regression case: given a dataset, fit a curve to it. Structure by multiple blocks of the same computation (the inductive prior) that we are trying to search for.
