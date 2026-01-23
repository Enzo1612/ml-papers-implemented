# Perceptron (Rosenblatt, 1958)

## Objective

Minimal implementation of the 1958 Rosenblatt Perceptron algorithm for binary classification of linearly separable data.

## Mathematical Model

**Decision Rule:**
$$f(x) = \begin{cases} 1 & \text{if } w \cdot x + b > 0 \\ -1 & \text{otherwise} \end{cases}$$

**Update Rule:**
For a misclassified point $(x, y)$:
$$w \leftarrow w + \eta \cdot y \cdot x$$
$$b \leftarrow b + \eta \cdot y$$

## Random Generator

Since we're only focusing on the learning part of the experiment, we have to generate random data to replicate the retina treating images.

### Step 1

Pick a random line that separates the universe.

- `w` a random vector `[x, y]` with $-1 \leq x, y \leq 1$.
- `b` a random bias with $-1 \leq b \leq 1$.

### Step 2

Generate random points.

$X = \text{random numbers between -1 and 1}$

### Step 3

Label the points based on the decision rule. Don't add points too close to 0, we need linearly separable data.
