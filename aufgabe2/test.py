import numpy as np


def matmult(A, B):
    
    dimA=A.shape
    dimB=B.shape
    
    m, n1, n2, p  = dimA[0], dimA[1], dimB[0], dimB[1]
    print(m)
    assert (n1 == n2),"Wrong Dimensions: Matrix Multiplication is not defined"
    C = np.zeros(m*p).reshape(m, p)
# Richtung m
    for i in range(m):
# Richtung p
        for k in range(p):
# Addition der Produkte
            for j in range(n1):
                print(C)
                C[i,k]+= A[i,j]*B[j,k]
    return print('\n', C)


M1 = np.array([[1, 2],
               [3, 4],
               [5, 6]])

M2 = np.array([[2, 0],
               [0, 2]])

print(M1, '\n', M2)

M_res = matmult(M1, M2)
print(M_res)