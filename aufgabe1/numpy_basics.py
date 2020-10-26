import numpy as np

# defining random number generator
rng = np.random.default_rng()

# (*) Erzeugen Sie einen Vektor mit Nullen der Länge 10 und setzen den Wert des 5. Elementes auf eine 1.
# YOUR CODE HERE
a = np.zeros(10)
a[5] = 1
print("a: \n", a)

# (*) Erzeugen Sie einen Vektor mit Ganzzahl-Werten von 10 bis 49 (geht in einer Zeile).
# YOUR CODE HERE
b = np.arange(10, 50)
print("\nb: \n", b)

# (*) Erzeugen Sie einen Vektor mit 8 Einträgen zwischen -1 und 1 bei dem alle Werte die gleichen Abstände habe und sowohl -1 als auch 1 enthalten sind (geht in einer Zeile). Hinweis: guckt euch den Unterschied zwischen numpy.arange und numpy.linspace welche Funktion passt besser?.
# YOUR CODE HERE
c = np.linspace(-1, 1, num=8)
print("\nc: \n", c)

# (*) Geben Sie nur das Stück (slice) von Vektor b) aus, das die Zahlen 21 bis 38 (Stellen 11 bis 28) beinhaltet (geht in einer Zeile).
# YOUR CODE HERE
d = b[11:29]
print("\nd: \n", d)

# (*) Ändern Sie den Vektor b) indem sie das Stück (slice) von Stelle 15 bis einschließlich Stelle 25 mit den Werten negierten Werten von Stelle 1 bis einschließlich Stelle 11 überschreiben (geht in einer Zeile).
# YOUR CODE HERE
print("\nb[15:26]: \n", b[15:26])
b[15:26] = (-1) * b[1:12]
print("\nb[15:26]: \n", b[15:26])

# (*) Drehen Sie die Werte des Vektors aus a) oder b) um (geht in einer Zeile).
# YOUR CODE HERE
print("\nf: \n", b)
f = np.flip(b)
print("\nf: \n", f)

# (*) Summieren Sie alle Werte in einem Array.
# YOUR CODE HERE
g = np.sum(b)
print("\ng: \n", g)

# (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 (links oben) bis 15 (rechts unten) (geht in einer Zeile). Tip: Schauen Sie sich numpy.reshape an
# YOUR CODE HERE
h = np.arange(16).reshape(4, 4)
print("\nh: \n", h)

# (*) Erzeugen Sie eine 5x3 Matrix mit Zufallswerteintegers zwischen 0-100 (geht in einer Zeile).
# YOUR CODE HERE
i = rng.integers(101, size=(5, 3))
print("\ni: \n", i)

# (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix (geht zwar in einer Zeile, aber benutzen Sie lieber Hilfsvariablen und drei Zeilen).
# YOUR CODE HERE
j1 = np.arange(12).reshape(4, 3)
j2 = np.arange(6).reshape(3, 2)
j = np.matmul(j1, j2)
print("\nj: \n", j)

# (*) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile aus (geht jeweils in einer Zeile).
# YOUR CODE HERE
k = np.arange(25).reshape(5, 5)
k_odd = k[::2]
k_even = k[1::2]
print("\nk_odd: \n", k_odd)
print("\nk_even: \n", k_even)


# (**) Erzeuge eine 5x5 Matrix mit Zufallsintegers zwischen 0-100 und finde deren Maximum und Minimum und normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - ein Wert wird 1 (max) sein und einer 0 (min)).
# Hinweis: D.h. Sie muessen die Werte normalisieren (R - R_min) / (R_max - R_min)
# YOUR CODE HERE
l = rng.integers(101, size=(5, 5))
l_nor = (l - np.min(l)) / (np.max(l) - np.min(l))
print("\nl: \n", l)
print("\nl: \n", l_nor)


# (**) Extrahieren Sie den Integer-Anteil eine Arrays von zufälliger Zahlen zwischen 0-10 auf 3 verschiedene Arten.
# YOUR CODE HERE

# (**) Erzeugen Sie eine Matrix $M$ der Größe 4x3 und einen Vektor $v$ mit Länge 3. Multiplizieren Sie jeden Spalteneintrag aus $v$ mit der kompletten Spalte aus $M$. Nutzen Sie dafür Broadcasting.
# YOUR CODE HERE
M = np.arange(12).reshape(4, 3)
v = np.arange(3).reshape(3)
M = M * v
print("\nM: \n", M)

# (***) Erzeugen Sie einen Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]).
# Konvertieren Sie diese in Polarkoordinaten \url{https://de.wikipedia.org/wiki/Polarkoordinaten}.
# Hinweis: nutzen Sie fuer die Berechnung des Winkel np.arctan2 und geben Sie jeweils Radius und Winkel als Vektor aus
# YOUR CODE HERE
o_kar = rng.integers(101, size=(10, 2))
r = np.sqrt(o_kar[:, 1::2]**2 + o_kar[:, ::2]**2)
angle = np.arctan2(o_kar[:, 1::2], o_kar[:, ::2]) * 180 / np.pi
o_pol = np.concatenate([r, angle], axis=1)

print("\no: \n", o_kar)
print("\no: \n", o_pol)

# (***) Erzeugen Sie einen Matrix der Größe 6x2, die Sie als Kartesische Koordinaten interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]).
# Schreiben Sie eine Funktion, die alle Punkt-Punkt Abstände berechnet.
# YOUR CODE HERE