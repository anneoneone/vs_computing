import numpy as np
import math
from numpy.linalg import inv


# (a) Berechnen Sie den Winkel $\alpha$ in Grad zwischen den folgenden beiden Vektoren $a=[1.,1.77]$ und $b=[1.5,1.5]$
print('\n-----------------AUFGABE 2A-----------------------\n')
a = np.array([-1.,1.77])
b = np.array([1.5,1.5])
# YOUR CODE HERE
def angle_between(a, b):
    dot_pr = a.dot(b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
 
    return np.rad2deg(np.arccos(dot_pr / norms))

print(angle_between(a,b))


# (b) Gegeben ist die quadratische regulaere Matrix A und ein Ergbnisvektor b. Rechnen Sie unter Nutzung der Inversen die Loesung x des Gleichungssystems Ax = b aus.
# YOUR CODE HERE
print('\n-----------------AUFGABE 2B-----------------------\n')
A = np.array([[2, 3, 4], [3, -2, -1], [5, 4 ,3]])
b = np.array([[1.4], [1.2], [1.4]])

A_inv = inv(A)

x_umstaendlich = np.matmul(A_inv, b)
x_einfach = np.linalg.solve(A, b)

print(x_umstaendlich)
print(x_einfach)

# Hinweis: Fangen Sie bitte mögliche falsche Eingabegroessen in der Funktion ab und werfen einen AssertionError
# assert Expression[, Arguments]
print('\n-----------------AUFGABE 2C-----------------------\n')


# are dimensions right?
def rightDimension(M1, M2):
    yM1, xM1 = M1.shape
    yM2, xM2 = M2.shape

    assert (xM1 == yM2), "Usage: xM1 should be yM2"

    return yM1, xM1, yM2, xM2

def matmult(M1, M2):
    yM1, xM1, yM2, xM2 = rightDimension(M1, M2)

    # create result matrix
    result = np.zeros(yM1 * xM2).reshape(yM1, xM2)

    # multiplicate the numbers
    for i in range(yM1):
        for k in range(xM2):
            for j in range(xM1):
                result[i,k]+= M1[i,j]*M2[j,k]
    return result


M1 = np.matrix('1 2; 3 4; 5 6')
M2 = np.matrix('2 0; 0 2')
print("M1", M1)
print("M2", M2)

M_res = matmult(M1,M2)
print("M_res", M_res)



