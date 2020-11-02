import numpy as np
import matplotlib.pyplot as plt



# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.
def create_matrix(x_ax, degree):
    A = np.ones((len(x_ax), degree + 1))

    for j in range(0, len(x_ax)):
        for i in range(0, degree):
            A[j][i] = x_ax[j]**(degree-i)

    return A


# Matrix nach x lösen
def solve_x(A, b):
    AtA = A.T.dot(A)
    Atb = A.T.dot(b)

    x = np.linalg.solve(AtA, Atb)

    return x


# Funktion generieren
def make_fkt(x):
    fkt = np.poly1d(x)

    return fkt


# Fehler e berechnen
def get_error(b, fkt, x_ax):
    k = 0
    e = 0

    for i in x_ax:
        e += abs(b[k] - fkt(i))
        k = k + 1

    return e


# Kleinste gemeinsame Quadrate berechnen und Fehler ausgeben
def least_square(x_ax, b, max_degree):
    # config
    e = np.zeros(max_degree)

    # Vorgang für alle Dimensionen durchführen
    for i in range(1,max_degree+1):
        # Matrix erstellen
        A = create_matrix(x_ax, i)
        # print("A: \n", A)

        # Ausgleichsproblem lösen
        x = solve_x(A, b)
        # print("x: \n", x)

        # Funktion ermitteln
        fkt = make_fkt(x)
        # print("fkt: \n", fkt)

        # Fehler ermitteln
        e[i-1] = get_error(b, fkt, x_ax)
        # print("err: \n", err)

    print(e)

    return fkt, e


# Definieren der x-Achse
x_ax = np.linspace(-2, 2, 200)
# print("x_ax: \n", x_ax)

# Laden der gegebenen Daten d0 - d4
b_a = np.load("data/d0.npy")
b_b = np.load("data/d1.npy")
b_c = np.load("data/d2.npy")
b_d = np.load("data/d3.npy")
b_e = np.load("data/d4.npy")

fkt_a, e_a = least_square(x_ax, b_a, 20)
fkt_b, e_b = least_square(x_ax, b_b, 20)
fkt_c, e_c = least_square(x_ax, b_c, 20)
fkt_d, e_d = least_square(x_ax, b_d, 20)
fkt_e, e_e = least_square(x_ax, b_e, 20)


# Plotten der Ergebnisse
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 7))

# d0.npy
fkt_a, e_a = least_square(x_ax, b_a, 1)
ax[0].scatter(x_ax, b_a, s=1)
ax[0].plot(x_ax, fkt_a(x_ax),'r')
ax[0].set_title("d0.npy")

# d1.npy
fkt_b, e_b = least_square(x_ax, b_b, 2)
ax[1].scatter(x_ax, b_b, s=1)
ax[1].plot(x_ax, fkt_b(x_ax),'r')
ax[1].set_title("d1.npy")

# d2.npy
fkt_c, e_c = least_square(x_ax, b_c, 4)
ax[2].scatter(x_ax, b_c, s=1)
ax[2].plot(x_ax, fkt_c(x_ax),'r')
ax[2].set_title("d2.npy")

# d3.npy
fkt_d, e_d = least_square(x_ax, b_d, 5)
ax[3].scatter(x_ax, b_d, s=1)
ax[3].plot(x_ax, fkt_d(x_ax),'r')
ax[3].set_title("d3.npy")

# d4.npy
fkt_e, e_e = least_square(x_ax, b_e, 7)
ax[4].scatter(x_ax, b_e, s=1)
ax[4].plot(x_ax, fkt_e(x_ax),'r')
ax[4].set_title("d4.npy")


plt.savefig('poly_fig.png', bbox_inches='tight')
plt.show()

