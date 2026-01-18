# Legendre's Least Squares Method

This project implements ideas from the appendix of the paper [Nouvelles méthodes pour la détermination des orbites des comètes par A.-M Legendre](https://gallica.bnf.fr/ark:/12148/bpt6k15209372/f84.item). Legendre introduced Least Squares as a fitting method that is still used today (e.g. **Ordinary Least Squares**).

**Normal Equation:**

$\min_x \lVert Ax - b\rVert^2 \;\Rightarrow\; A^T A x = A^T b$

## Concept

From a list of observations (comet observations originally), the objective is to fit a model. When the number of observations exceeds the number of unknowns, an exact solution might not exist. The objective is to find the line that minimizes the sum of squared errors.
