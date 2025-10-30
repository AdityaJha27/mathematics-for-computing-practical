# mathematics-for-computing-practical
# 1. Creating Vectors and Matrices using NumPy
    
    import numpy as np
    
    # Horizontal Vector
    hv = np.array([1, 2, 3])
    print("Horizontal Vector:\n", hv)
    
    # Vertical Vector
    vv = np.array([1, 2, 3]).reshape(-1, 1)
    print("\nVertical Vector:\n", vv)
    
    # Matrix from list
    A = np.array([[1, 2, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    print("\nMatrix A:\n", A)
    
    # Matrix using np.matrix
    B = np.matrix([[1, 2, 0],
                   [0, 0, 0],
                   [0, 0, 1]])
    print("\nMatrix B (np.matrix):\n", B)

# 2. Matrix Transpose and Rank using NumPy

    import numpy as np
    
    # Taking matrix input
    s = input("Enter matrix rows separated by ';' e.g. 1 2 0; 0 0 0; 0 0 1\n> ")
    rows = [r.strip() for r in s.split(';')]
    A = np.array([list(map(float, r.split())) for r in rows])
    
    print("\nMatrix A:\n", A)
    print("\nTranspose of A:\n", A.T)
    print("\nRank of A:", np.linalg.matrix_rank(A))

# 3. Minor, Cofactor and Adjugate of a Matrix

    import numpy as np
    
    def minor(A, i, j):
        M = np.delete(np.delete(A, i, axis=0), j, axis=1)
        return round(np.linalg.det(M))
    
    def cofactor_matrix(A):
        C = np.zeros_like(A)
        for i in range(len(A)):
            for j in range(len(A)):
                C[i, j] = ((-1)**(i+j)) * minor(A, i, j)
        return C
    
    # Matrix
    A = np.array([[1, 2, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    
    C = cofactor_matrix(A)
    print("Matrix A:\n", A)
    print("\nCofactor Matrix:\n", C)
    print("\nAdjugate Matrix:\n", C.T)

# 4. Solving Linear Equations using Gauss-Jordan (NumPy)

    import numpy as np
    
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]])
    
    B = np.array([8, -11, -3])
    
    # Solve Ax = B
    x = np.linalg.solve(A, B)
    print("Solution:\n", x)

# 5. Null Space and Rank-Nullity Theorem

    import numpy as np
    from scipy.linalg import null_space
    
    A = np.array([[1, 2, 0],
                  [0, 0, 0],
                  [0, 0, 1]], dtype=float)
    
    print("Matrix A:\n", A)
    
    ns = null_space(A)
    print("\nNull Space:\n", np.round(ns, 4))
    
    rank = np.linalg.matrix_rank(A)
    nullity = A.shape[1] - rank
    
    print("\nRank =", rank)
    print("Nullity =", nullity)
# 6. Elementary Row Operations (Row Space)

    import numpy as np
    
    A = np.array([[1, 2, 0],
                  [0, 0, 0],
                  [0, 0, 1]], dtype=float)
    
    print("Original Matrix:\n", A)
    
    A[1] = A[1] - 2*A[0]
    A[2] = A[2] - 3*A[0]
    
    print("\nAfter Row Operations:\n", A)

# 7. Determinant of a Matrix using NumPy

    import numpy as np
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    detA = np.linalg.det(A)
    
    print("Matrix A:\n", A)
    print("\nDeterminant of A =", round(detA, 2))

# 8. Inverse of a Matrix using NumPy

    import numpy as np
    
    A = np.array([[1, 2],
                  [3, 4]])
    
    invA = np.linalg.inv(A)
    
    print("Matrix A:\n", A)
    print("\nInverse of A:\n", invA)

# 9. Eigenvalues and Eigenvectors using NumPy

    import numpy as np
    
    A = np.array([[2, 1],
                  [1, 2]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Matrix A:\n", A)
    print("\nEigenvalues:\n", eigenvalues)
    print("\nEigenvectors:\n", eigenvectors)

# 10. Solve Homogeneous System using Gauss-Jordan Method
    
    import sympy as sp
    
    A = sp.Matrix([[2, 1, -1],
                   [1, -3, 2],
                   [3, -1, 1]])
    
    print("Matrix A:\n", A)
    
    rref_matrix, pivots = A.rref()
    
    print("\nReduced Row Echelon Form:")
    print(rref_matrix)
    
    print("\nSolution of Homogeneous System:")
    print(A.nullspace())

# 11. Matrix Multiplication using NumPy

    import numpy as np
    
    A = np.array([[1, 2],
                  [3, 4]])
    
    B = np.array([[2, 0],
                  [1, 2]])
    
    C = np.dot(A, B)
    
    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    print("\nA × B =\n", C)

# 12. Trace of a Matrix using NumPy
    # (Sum of diagonal elements)
    
    import numpy as np
    
    A = np.array([[4, 2, 1],
                  [0, 3, 5],
                  [7, 8, 6]])
    
    traceA = np.trace(A)
    
    print("Matrix A:\n", A)
    print("\nTrace of A =", traceA)




      




 
