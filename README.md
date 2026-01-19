# Machine Learning Papers Implemented

## Description

Reimplementing influential machine learning papers from first principles.

> What I cannot create, I do not understand - R. Feynman

This repository contains minimal, dependency-free implementations of historically important machine learning and optimization papers. The objective is not performance but understanding.

## Scope and Intent

Modern ML stacks abstract away the mathematics that actually does the work.

### Design constraints:

- Pure python only
- Explicit linear algebra
- Algorithms follow the original derivations
  The resulting code might be slower but transparent and faithful.

## Implemented Papers

| **Year** | **Paper**                                                                                                | **Author** | **Focus**                       |
| -------- | -------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------- |
| 1805     | [Nouvelles méthodes pour la détermination des orbites des comètes](./01_least_squares_legendre)          | Legendre   | Least Squares, Normal Equations |
| 1847     | [Méthode générale pour la résolution des systèmes d'équations sumultanées](./02_gradient_descent_cauchy) | Cauchy     | Optimization, Gradient Descent  |
| ...      | ...                                                                                                      | ...        | ...                             |

Papers are implemented in **historical order** to track how ideas evolved into modern ML.

## Mathematical Core

To enforce first-principle reasoning, all numerical machinery lives in a custom linear algebra layer. The repository containing the full library is available on my account.

Implemented components include:

- Vector spaces (dot product, norms)
- Dense matrices
- Gaussian elimination
- Back-substitution
- Basic linear solvers

## Running the Code

No dependencies required. Each paper directory is self-contained and runnable.

## Philosophy

Modern AI systems reduce cognitive load by design. While this is powerful, it also removes many of the failure modes through which understanding is forged. This repository resists that erosion by controlling its role.

AI is used here to:

- Verify results
- Give me hints
- Critique and refine the code, docstrings and documentation

It is not used to generate solutions end-to-end. I'm still learning and often require external help.
