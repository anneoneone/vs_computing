

import numpy as np
from math import sqrt, fabs, pi
import matplotlib.pyplot as plt

# Aufgabe 1 (a)
# i ist hier die Anzahl der Iterationen
# In jeder Iteration soll ein epsilon auf 1.0 addiert werden und mit der
# Floating-Point Darstellung von np.float64(1) bzw. np.float(32) verglichen werden.
# Starten Sie dabei mit Epsilon=1.0 und halbieren Sie den Wert in jeder Iteration (wie an der Ausgabe 2^(-i) zu sehen)
# Stoppen Sie die Iterationen, wenn np.float32(1) + epsi != np.float32(1) ist.
# Hinweis Ja  in diesem Fall dürfen Sie Floating-Point Werte vergleichen 

# Print Anweisung vor dem Loop
print('\n-----------------AUFGABE 1A-----------------------\n')
print('i | 2^(-i) | 1 + 2^(-i) ')
print('----------------------------------------')

##
#  YOUR CODE HERE
##

epsi = float(1)
i = 0

while float(1) != float(1) + epsi:
    i += 1
    epsi = float(2**(-i))

    pass


# Print Anweisung in / nach dem Loop
print('float: {0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi))



epsi32 = np.float32(1)
i = 0

while np.float32(1) != np.float32(1) + epsi32:
    i += 1
    epsi32 = np.float32(2**(-i))

    pass


# Print Anweisung in / nach dem Loop
print('float32: {0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi32))


epsi64 = np.float64(1)
i = 0

while np.float64(1) != np.float64(1) + epsi64:
    i += 1
    epsi64 = np.float64(2**(-i))

    pass


# Print Anweisung in / nach dem Loop
print('float64: {0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi64))



# Aufgabe 1 (b)
print('\n-----------------AUFGABE 1B-----------------------\n')
# Werten Sie 30 Iterationen aus und speichern Sie den Fehler in einem
# Fehlerarray err
N = 30
err = []
# sqrt(2) kann vorberechnet werden
sn = sqrt(2)
U_real = 2 * pi

for n in range(2, N):
    # 1. Umfang u berechnen
    U_sn = 2**n * sn

    sn = sqrt(2 - sqrt(4 - sn**2))

    # 2. Fehler en berechnen und in err speichern
    # Fehler ausgeben print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))
    # YOUR CODE HERE
    en = fabs(U_real - U_sn)
    err.append(en)

    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, U_sn, en))


# Plotten Sie den Fehler
plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)


# Aufgabe 1 (c)
print('\n-----------------AUFGABE 1C-----------------------\n')
# Löschen des Arrays und wir fangen mit der Berechnung von vorn an.
# Nur diesmal mit der leicht veranderten Variante
err = []
sn = sqrt(2)
U_real = 2 * pi

for n in range(2, N):
    # 1. Umfang u berechnen
    U_sn = 2**n * sn

    sn = sn/(sqrt(2 + sqrt(4 - sn**2)))

    # 2. Fehler en berechnen und in err speichern
    # Fehler ausgeben print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))
    # YOUR CODE HERE
    en = fabs(U_real - U_sn)
    err.append(en)

    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, U_sn, en))

plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()


