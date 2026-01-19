from math import sqrt
from typing import List


class Vector:
    """A mathematical vector with basic operations.
    
    Supports addition, subtraction, scalar multiplication, dot product,
    and norm calculations.
    """
    
    def __init__(self, coords:List[float]):
        """Initialize a vector with coordinates.
        
        Args:
            coords (List[float]): List of coordinate values.
        """
        self._coor = list(coords)

    @property
    def coords(self) -> List[float]:
        """Get the vector coordinates.
        
        Returns:
            List[float]: The coordinate values.
        """
        return self._coor
    
    def check_len(self, v2: 'Vector'):
        """
        Verifies that the argument has the same dimension as self

        Args:
            v2 (Vector): The vector to compare self to.

        Raises:
            ValueError: If dimensions do not match
        """
        if len(self) != len(v2):
            raise ValueError("Vectors must have the same dimension")
        
    def __repr__(self):
        """
        toString function to print the vector in the right format.

        Returns: A string representing the vector coordinates.
        """
        return f"{self.coords}"
        
    def __eq__(self, v2: 'Vector'):
        """
        Checks if two vectors are element-wise equal.
        Args: 
            v2 (Vector): The vector from which we want to compare coordinates with self's coordinates.
        """
        if not isinstance(v2, Vector):
            return NotImplemented
        return self.coords == v2.coords

    def __add__(self, v2: 'Vector') -> 'Vector':
        """ 
        Computes the element-wise sum of two vectors.

        Args:
            v2 (Vector): The vector to add. Must have same dimensions.
        Raises:
            ValueError: If dimensions are not the sames.
        """
        self.check_len(v2)
        return Vector([a + b for a, b in zip(self.coords, v2.coords)])

    def __sub__(self, v2:'Vector') -> 'Vector':
        """
        Computes the element-wise subtraction of two vectors.

        Args:
            v2 (Vector): The vector to subtract. Must have same dimensions.
        Raises:
            ValueError: If dimensions are not the same.
        """
        self.check_len(v2)
        return Vector([a - b for a, b in zip(self.coords, v2.coords)])

    def __mul__(self, scalar: float) -> 'Vector':
        """
        Computes the element-wise multiplication of a scalar and a vector.

        Args:
            scalar (float)
        """
        return Vector([a * scalar for a in self.coords])

    def __rmul__(self, scalar:float) -> 'Vector':
        """
        Computes the reverse multiplication of a vector and a scalar.

        Args:
            scalar (float)
        """
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar:float) -> 'Vector':
        """
        Computes the element-wise true division of a vector by a scalar.

        Args:
            scalar(float)
        
        Raises:
            ValueError: If scalar = 0
        """
        if scalar == 0:
            raise ValueError("Can't divide by 0!")
        return Vector([a / scalar for a in self.coords])

    def __len__(self):
        """
        Returns the length of the Vector coordinates.
        """
        return (len(self._coor))
    
    def __getitem__(self, i:int):
        """
        Returns item i of the coordinates of the Vector.
        """
        return self._coor[i]
    
    def __setitem__(self, i:int, coords:float):
        """Set the value at index i.
        
        Args:
            i (int): The index to set.
            coords (float): The new value.
        """
        self._coor[i] = coords

    def dot(self, v2: 'Vector') -> float:
        """
        Computes the dot product of two vectors.

        Args:
            v2 (Vector): The vector with which the dot product is performed.
        Raises:
            ValueError: If dimensions are not the same.
        """
        self.check_len(v2)
        return sum(a * b for a, b in zip(self.coords, v2.coords))
    
    def copy(self) -> 'Vector':
        return Vector(self.coords[:])

    @property
    def norm(self) -> float:
        """
        Returns the norm of a vector (square root of the dot product of the vector and itself).
        """
        return sqrt(self.dot(self))