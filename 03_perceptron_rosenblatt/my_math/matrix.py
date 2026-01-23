from typing import List, Optional, overload
from src.vector import Vector


class Matrix:
    """A mathematical matrix with basic operations.
    
    Supports addition, subtraction, scalar multiplication, and matrix-vector
    and matrix-matrix multiplication.
    """
    
    def __init__(self, rows: List):
        """Initialize a matrix from a list of rows.
        
        Args:
            rows (List): List of lists representing matrix rows. All rows must have the same length.
            
        Raises:
            ValueError: If matrix is empty or rows have inconsistent lengths.
        """
        if not rows or not rows[0]:
            raise ValueError("Matrix can't be empty")
        
        length = len(rows[0])
        for row in rows:
            if len(row) != length:
                raise ValueError("All rows must have the same length")
            
        self._rows = [Vector(row) for row in rows]

        self._P = None
        self._L = None
        self._U = None

        self._inverse = None
    
    def apply_func(self, f):
        return Matrix([[f(x) for x in self.rows]])

    def __eq__(self, b: 'Matrix') -> bool:
        """
        Check if two matrices are equal element-wise.

        Args:
            b (Matrix): The matrix to compare with

        Returns:
            bool: True if matrices have same shape and all elements are equals. Else return False.
        """
        if self.shape != b.shape:
            return False
        return all(row_a == row_b for row_a, row_b in zip(self.rows, b.rows))
    

    def row(self, i: int) -> Vector:
        """
        Returns the i^th row of the matrix.

        Args:
            i (int): The index of the row.
        """
        return self.rows[i]
    
    def col(self, j: int) -> Vector:
        """
        Returns the j^th column of the matrix.
        
        Args:
            j (int): The index of the column.
        """
        return Vector([row[j] for row in self.rows])
    
    @overload
    def __matmul__(self, x:Vector) -> Vector: ...

    @overload
    def __matmul__(self, x:'Matrix') -> 'Matrix': ...
    
    def __matmul__(self, x):
        """
        Matrix multiplication.

        Supports:
        - Matrix–Vector: row-wise dot with `x` (length must equal number of columns).
        - Matrix–Matrix: classical product where `(A @ B)[i, j] = row_i(A) · col_j(B)`.

        Args:
            x (Vector | Matrix): The right-hand operand.

        Raises:
            ValueError: If `x` is a `Vector` and its length differs from the matrix's column count.
            ValueError: If `x` is a `Matrix` and A's column count != B's row count.
            TypeError: If `x` is neither `Vector` nor `Matrix`.
        """
        if isinstance(x, Vector):
            m, n = self.shape
            if n != len(x):
                raise ValueError("The vector length must be the same as the matrix's number of cols")
            res = [row.dot(x) for row in self.rows]
            return Vector(res)
        
        if isinstance(x, Matrix):
            m, n = self.shape
            if n != x.shape[0]:
                raise ValueError("The matrix 1 column dimension must match the matrix 2 row dimension")
            else:
                new_rows = []
                for row_a in self.rows:
                    current_new_row_data = []

                    for j in range(x.shape[1]):
                        current_new_row_data.append(row_a.dot(x.col(j)))
                    new_rows.append(current_new_row_data)
                return Matrix(new_rows)
        else:
            raise TypeError(f"Cannot multiply Matrix by {type(x)}")
    
    def check_dim(self, b: 'Matrix'):
        """
        Checks if the matrices have the same dimensions.

        Raises:
            ValueError: If the matrices don't have the same dimensions.
        """
        if self.shape != b.shape:
            raise ValueError("The matrices don't have the same dimensions")
        
    def __getitem__(self, i:int) -> Vector:
        """Get the i-th row of the matrix.
        
        Args:
            i (int): The row index.
            
        Returns:
            Vector: The i-th row as a Vector.
        """
        return self.rows[i]
           
    @overload
    def __add__(self, b: float) -> 'Matrix': ...
    
    @overload
    def __add__(self, b: Vector) -> 'Matrix': ...
    
    @overload
    def __add__(self, b: 'Matrix') -> 'Matrix': ...
    
    def __add__(self, b):
        """
        Computes the sum of matrix with a scalar, vector, or another matrix.
        
        Args:
            b (float | Vector | Matrix): The value to add.
                - float: Adds scalar to every element
                - Vector: Converts to row matrix and adds element-wise
                - Matrix: Adds element-wise
        
        Raises:
            ValueError: If dimensions don't match for vector/matrix addition.
            TypeError: If b is not a supported type.
        """
        if isinstance(b, (int, float)):
            return Matrix([row + b for row in self.rows])
        
        if isinstance(b, Vector):
            # Convert vector to a 1xn matrix
            b_matrix = Matrix([b.coords])
            self.check_dim(b_matrix)
            return Matrix([row_a + row_b for row_a, row_b in zip(self.rows, b_matrix.rows)])
        
        if isinstance(b, Matrix):
            self.check_dim(b)
            return Matrix([row_a + row_b for row_a, row_b in zip(self.rows, b.rows)])
        
        raise TypeError(f"Cannot add Matrix with {type(b)}")
    
    @overload
    def __sub__(self, b: float) -> 'Matrix': ...
    
    @overload
    def __sub__(self, b: Vector) -> 'Matrix': ...
    
    @overload
    def __sub__(self, b: 'Matrix') -> 'Matrix': ...
    
    def __sub__(self, b):
        """
        Computes the subtraction of a scalar, vector, or matrix from the matrix.
        
        Args:
            b (float | Vector | Matrix): The value to subtract.
                - float: Subtracts scalar from every element
                - Vector: Converts to row matrix and subtracts element-wise
                - Matrix: Subtracts element-wise
        
        Raises:
            ValueError: If dimensions don't match for vector/matrix subtraction.
            TypeError: If b is not a supported type.
        """
        if isinstance(b, (int, float)):
            return Matrix([row - b for row in self.rows])
        
        if isinstance(b, Vector):
            # Convert vector to a 1xn matrix
            b_matrix = Matrix([b.coords])
            self.check_dim(b_matrix)
            return Matrix([row_a - row_b for row_a, row_b in zip(self.rows, b_matrix.rows)])
        
        if isinstance(b, Matrix):
            self.check_dim(b)
            return Matrix([row_a - row_b for row_a, row_b in zip(self.rows, b.rows)])
        
        raise TypeError(f"Cannot subtract {type(b)} from Matrix")
    
    def __mul__(self, scalar: float) -> 'Matrix':
        """Multiply the matrix by a scalar.
        
        Args:
            scalar (float): The scalar value to multiply by.
            
        Returns:
            Matrix: A new matrix with all elements multiplied by the scalar.
        """
        return Matrix([row * scalar for row in self.rows])
    
    def __rmul__(self, scalar: float) -> 'Matrix':
        """Support reverse multiplication (scalar * matrix).
        
        Args:
            scalar (float): The scalar value to multiply by.
            
        Returns:
            Matrix: A new matrix with all elements multiplied by the scalar.
        """
        return self * scalar
    
    def __setitem__(self, target:int, new_row: Vector):
        self._rows[target] = new_row
    
    def subtract_row(self, source: int, target: int, mult: float):
        self[target] = self[target] - self[source] * mult

    def gaussian_elimination(self, b: Optional[Vector] = None) -> int:
        m, n = self.shape
        swap_count = 0
        for j in range(min(m, n)):
            pivot_row = j
            max_val = 0.0

            for i in range(j, m):
                val = abs(self[i][j])
                if val > max_val:
                    max_val = val
                    pivot_row = i
                
            if max_val < 1e-10:
                raise ValueError("Matrix is singular. 0 pivot found everywhere.")
            
            if pivot_row != j:
                self.swap_rows(pivot_row, j, b)
                swap_count += 1
            for i in range(j+1, m):
                mult = self[i][j] / self[j][j]
                self.subtract_row(j, i, mult)
                if b != None:
                    b[i] = b[i] - (mult * b[j])
        return swap_count
    
    def back_substitution(self, b: Vector) -> Vector:
        m, n = self.shape

        x_data = [0.0] * n

        for i in range(n-1, -1, -1):
            sum_ax = b[i]
            for j in range(i+1, n):
                sum_ax -= self[i][j] * x_data[j]
            if self[i][i] == 0:
                raise ValueError("Singular matrix found during backsubstitution.")
            x_data[i] = sum_ax / self[i][i]
        return Vector(x_data)
    
    def solve(self, b: Vector) -> Vector:
        """
        Solves the system Ax = b.

        Args:
            b (Vector): The RHS of the equation
        Raises:
            ValueError: If a 0 pivot is found
        """
        self.gaussian_elimination(b)

        return self.back_substitution(b)
    
    def copy(self) -> 'Matrix':
        return Matrix([row.coords[:] for row in self.rows])
    
    
    def swap_rows(self, row1: int, row2: int, b: Optional[Vector] = None):
        """
        Swaps row1 and row2 in the matrix.
        If a vector b is provided, swaps the corresponding elements in b as well.
        """
        self[row1], self[row2] = self[row2], self[row1]
        if b != None:
            b[row1], b[row2] = b[row2], b[row1]

    def plu_decomposition(self) -> tuple['Matrix', 'Matrix', 'Matrix']:
        """
        Decomposes the matrix A into PA = LU (LU Factorization with Partial Pivoting).

        Returns:
            P (Matrix): The permutation matrix recording row swaps.
            L (Matrix): The lower triangular matrix of multipliers (with 1s on diagonal).
            U (Matrix): The upper triangular matrix (pivots on diagonal).
            
        Raises:
            ValueError: If the matrix is not square.
            ValueError: If the matrix is singular (all potential pivots in a column are 0).
            
        Invariant:
            P @ A == L @ U
        """
        if self._P is not None and self._U is not None and self._L is not None:
            return self._P, self._L, self._U
        m, n = self.shape
        if (m != n):
            raise ValueError("Matrix must be square!")
        U = self.copy()
        L = self.identity(n)
        P = self.identity(n)
        for j in range(n):
            pivot_row = j
            max_val = 0.0
            for k in range(j, n):
                val = abs(U[k][j])
                if val > max_val:
                    max_val = val
                    pivot_row = k
            if max_val == 0:
                raise ValueError("Matrix is singular")
        
            if j != pivot_row:
                U.swap_rows(j, pivot_row)
                P.swap_rows(j, pivot_row)

                for col in range(j):
                    L[j][col], L[pivot_row][col] = L[pivot_row][col], L[j][col]
            
            for i in range(j + 1, n):
                multiplier = U[i][j] / U[j][j]
                L[i][j] = multiplier
                U[i] = U[i] - U[j] * multiplier

        self._P = P
        self._U = U
        self._L = L
        return P, L, U
    
  
    
    
    def solve_plu(self, b: 'Vector') -> 'Vector':
        P, L, U = self.P, self.L, self.U
        Pb = P @ b
        m, n = self.shape
        y = [0.0] * n
        for j in range(n):
            y[j] = Pb[j]
            for i in range(j):
                y[j] -= L[j][i] * y[i]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= U[i][j] * x[j]

            if abs(U[i][i]) < 1e-10: # don't compare floats to 0, thanks programming
                raise ValueError("Singular Matrix")
            x[i] /= U[i][i]
        
        return Vector(x)
    
    def rref(self) -> 'Matrix':
        """
        Computes the Reduced Row Echelon Form (RREF) of the matrix.
        
        Uses Gaussian elimination with partial pivoting to transform the matrix
        into RREF, where:
        - Each pivot is 1
        - All entries above and below each pivot are 0
        - Each pivot is to the right of the pivot in the row above
        
        Returns:
            Matrix: A new matrix in reduced row echelon form.
            
        Note:
            This method does not modify the original matrix.
        """
        m, n = self.shape
        RREF = self.copy()
        pivot_row = 0
        col = 0
        while(pivot_row < m and col < n):
            max_val = 0.0
            max_row = -1

            for j in range(pivot_row, m):
                if abs(RREF[j][col]) > abs(max_val):
                    max_val = RREF[j][col]
                    max_row = j

            if abs(max_val) < 1e-10:
                col += 1
                continue
            if max_row != pivot_row:
                RREF.swap_rows(pivot_row, max_row)

            pivot_val = RREF[pivot_row][col]
            for j in range(n):
                RREF[pivot_row][j] /= pivot_val

            for i in range(m):
                if i != pivot_row:
                    factor = RREF[i][col]
                    if abs(factor) > 1e-10:
                        for k in range(col, n):
                            RREF[i][k] -= factor * RREF[pivot_row][k]

            pivot_row += 1
            col += 1
        return RREF
    
    @property
    def det(self):
        """
        Calculates the determinant using Gaussian Elimination.
        det(A) = product of pivots in upper triangular form
        """
        m, n = self.shape
        if m != n:
            raise ValueError("Determinant is only defined for square matrices.")
        u = self.copy()
        count = u.gaussian_elimination()
        det = 1
        for i in range(m):
            det *= u[i][i]
            
        return det * ((-1) ** count)
    
    @property
    def rows(self):
        return self._rows
    
    @property
    def shape(self):
        """
        Returns the size of the matrix as a tuple (number of rows, number of cols).
        """
        return (len(self.rows), len(self.rows[0]))
        
                
    
    @property
    def cols(self) -> List[Vector]:
        """
        Returns all columns as a list of Vectors.
        """

        return [self.col(j) for j in range(self.shape[1])]

    @property
    def inverse(self) -> 'Matrix':
        if self._inverse is not None:
            return self._inverse
        n = self.shape[1]
        inv_cols = []

        I = self.identity(n)
        for row in I.rows:
            inv_cols.append(self.solve_plu(row))

        inv_rows = []
        for i in range(n):
            inv_rows.append([inv_cols[j][i] for j in range(n)])

        self._inverse = Matrix(inv_rows)
        
        return self._inverse

    @property
    def P(self) -> 'Matrix':
        if self._P is None:
            self.plu_decomposition()
        return self._P # type: ignore
    
    @property
    def L(self) -> 'Matrix':
        if self._L is None:
            self.plu_decomposition()
        return self._L # type: ignore
    
    @property
    def U(self) -> 'Matrix':
        if self._U is None:
            self.plu_decomposition()
        return self._U # type: ignore
    
    @property
    def transpose(self) -> 'Matrix':
        """
        Returns the tranpose of the matrix
        """
        m, n = self.shape
        return Matrix([[self[i][j] for i in range(m)] for j in range(n)])



    @staticmethod
    def identity(n: int) -> 'Matrix':
        """
        Returns an identity matrix of size n
        """
        return Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]) 



        
