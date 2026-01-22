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
