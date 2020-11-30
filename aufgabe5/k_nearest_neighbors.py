import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import Counter


def distance(a, b):
    """calculates the distance between two vectors (or matrices)"""
    # 2.1.1 Berechnen Sie die Distanz zwischen zwei Matritzen/Bildern
    ...


def knn(query, data, labels, k):
    """
    Calculates the k-NN and returns the most common label appearing in the k-NN
    and the number of occurrences.
    For each data-record i the record consists of the datapoint (data[i]) and
    the corresponding label (label[i]).

    :param query: ndarray representing the query datapoint.
    :param data: list of datapoints represents the database together with labels.
    :param labels: list of labels represents the database together with data.
    :param k: Number of nearest neighbors to consider.
    :return: the label that occured the most under the k-NN and the number of occurrences.
    """
    # 2.1 Berechnen Sie die Distanzen von query zu allen Elementen in data
    # Implementieren Sie dazu die Funktion distance
    ...
    # 2.2 Finden Sie die k nächsten datenpunkte in data

    # 2.3 Geben Sie das Label, welches am häufigsten uner den k nächsten Nachbar
    # vorkommt und die Häufigkeit als tuple zurück.
    # Tipp: Counter(["a","b","c","b","b","d"]).most_common(1)[0]
    # returned das häufigste Element der Liste und deren Anzahl also ("b", 3)


# ---------------------------------------------------------------------------
# k Nearest Neighbors
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Bauen Sie die Datenbank auf. Sie besteht aus zwei Listen.
    # Einmal die Datenpunkte (die Bilder) und die dazugehörigen Label.
    # Die beiden Listen werden seperat gespeichert, aber gehören zusammen, dh.
    # Liste_der_Datenpunkte[i] gehört zu Liste_der_Labels[i].
    # Die Listen sind also gleich lang.
    # Tipp:
    # mit glob.glob("images/db/test/*") bekommen Sie eine Liste mit allen Dateien in dem angegebenen Verzeichnis

    img = plt.imread("images/db/test/car.jpg").copy()
    plt.imshow(img)
    plt.show()

    folder = glob.glob("images/db/train/*")
    list_datapoints = []    
    list_labels = []
    
    for i in folder:
        list_datapoints += glob.glob(i + "/*")

        for j in range(0, len(glob.glob(i + "/*"))):
            path, name = i.rsplit("/", 1)
            list_labels.append(name[:-1])

    print(len(list_datapoints))
    print(len(list_labels))
    
    
    # 2. Implementieren Sie die Funktion knn.
    Ele, Nr = knn(img, list_datapoints, list_labels, 5)

    # 3. Laden Sie die Testbilder aus dem Ordner "images/db/test/" und rufen Sie
    # auf der Datenbank knn auf. Geben Sie zu jedem Testbild das prognostizierte Label aus.
    # Varieren Sie den Parameter k.
    # Hinweis: Mit k = 5 sollte das beste Ergebnis erzielt werden.
