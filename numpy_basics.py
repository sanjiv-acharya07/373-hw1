import numpy as np
from random import randint



class NumpyBasics:
    """Class includes some numpy basic operations."""

    def create_identity_matrix(size: int):
        """Create an identity matrix of the given size.

        Args:
            size:
                Number of rows and columns in the matrix.

        Returns:
            A 2-D numpy identity matrix of the given size. 
        """
        # >> YOUR CODE HERE
        x = np.identity(size)
        return x
        # END OF YOUR CODE

    def create_column_vector(n: int):
        """Create a column vector of the given size with entries starting at 0 and ending at n-1.

        Args:
            n:
                Number of elements in the vector.

        Returns:
            A 2-D numpy column vector with shape (n, 1). 
        """
        # >> YOUR CODE HERE
        final = []
        for i in range(n):
            number = []
            number.append(i)
            final.append(number)
        c_vector = np.array(final)
        return c_vector
        # END OF YOUR CODE

    def calculate_trace(matrix: np.ndarray):
        """Calculate the trace of the given square matrix.

        Args:
            matrix:
                A 2-D numpy matrix.

        Returns:
            The sum of diagonal elements of the matrix.
        """
        # >> YOUR CODE HERE
        sums = np.trace(matrix)
        return sums
        # END OF YOUR CODE

    def multiply_then_sum(matrix1: np.ndarray, matrix2: np.ndarray):
        """Element-wise multiply two matrices and then sum along rows.

        Args:
            matrix1:
                A 2-D numpy matrix.
            matrix2:
                A 2-D numpy matrix of the same shape as matrix1.

        Returns:
            A 2-D numpy array with shape (m, 1) where each element is the sum 
            of the product of the corresponding row of matrix1 and matrix2.
        """
        # >> YOUR CODE HERE
        new_matrix = np.multiply(matrix1, matrix2)
        sum_matrix = np.sum(new_matrix, axis = 1)
        rows_list = sum_matrix.tolist()
        final = []
        for i in rows_list:
            number = []
            number.append(i)
            final.append(number)
        c_vector = np.array(final)
        
        return c_vector
        # END OF YOUR CODE

    def get_eigenvalues(matrix: np.ndarray):
        """Get eigenvalues of the given square matrix.

        Args:
            matrix:
                A 2-D numpy matrix.

        Returns:
            A 1-D numpy array containing the eigenvalues of the matrix.
        """
        # >> YOUR CODE HERE
        eigenvalue, eigenvector = np.linalg.eig(matrix)
        final = np.array(eigenvalue)
        return final
        # END OF YOUR CODE


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

from random import seed
import os

def test(func, expected_output, **kwargs) -> bool:
    """
    Test a function with some inputs.

    Args:
        func: The function to test.
        expected_output: The expected output of the function.
        **kwargs: The arguments to pass to the function.

    Returns:
        True if the function outputs the expected output, False otherwise.
    """
    output = func(**kwargs)

    try:
        assert np.allclose(output, expected_output)
        print(f'Testing {func.__name__}: passed')
        return True
    except AssertionError:
        print(f'Testing {func.__name__}: failed')
        print(f'Expected:\n {expected_output}')
        print(f'Got:\n {output}')
        return False
    
def evaluate_numpy_basics():
    """
    Test your implementation in numpy_basics.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Numpy Basics-------------\n')
    print('This test is not exhaustive by any means. You should test your ')
    print('implementation by yourself.\n')

    # Test create_identity_matrix
    test(NumpyBasics.create_identity_matrix, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), size=3)
    test(NumpyBasics.create_identity_matrix, np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]), size=5)

    # Test create_column_vector
    test(NumpyBasics.create_column_vector, np.array([[0], [1], [2], [3]]), n=4)
    test(NumpyBasics.create_column_vector, np.array([[0], [1], [2]]), n=3)

    # Test calculate_trace
    test(NumpyBasics.calculate_trace, 6, matrix=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
    test(NumpyBasics.calculate_trace, 15, matrix=np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]]))

    # Test multiply_then_sum
    test(NumpyBasics.multiply_then_sum, np.array([[4], [14], [33]]), 
         matrix1=np.array([[1, 2], [3, 4], [5, 6]]), matrix2=np.array([[2, 1], [2, 2], [3, 3]]))
    test(NumpyBasics.multiply_then_sum, np.array([[10], [30]]), 
         matrix1=np.array([[1, 2, 3], [4, 5, 6]]), matrix2=np.array([[2, 1, 2], [2, 2, 2]]))

    # Test get_eigenvalues
    test(lambda matrix: np.sort(NumpyBasics.get_eigenvalues(matrix)), np.array([1, 3]), matrix=np.array([[2, 1], [1, 2]]))
    test(lambda matrix: np.sort(NumpyBasics.get_eigenvalues(matrix)), np.array([2, 3]), matrix=np.array([[4, -1], [2, 1]]))


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_numpy_basics()
