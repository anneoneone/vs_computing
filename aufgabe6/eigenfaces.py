import numpy as np
import lib
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimage


####################################################################################################


def load_images(path: str) -> list:
    """
    Load all images in path

    :param path: path of directory containing image files

    :return images: list of images (each image as numpy.ndarray and dtype=float64)
    """
    # 1.1 Laden Sie für jedes Bild in dem Ordner das Bild als numpy.array und
    # speichern Sie es in einer "Datenbank" eine Liste.
    # Tipp: Mit glob.glob("data/train/*") bekommen Sie eine Liste mit allen
    # Dateien in dem angegebenen Verzeichnis.
    folder = glob.glob(path + "*")
    image_list = []

    # alle Bilder im Ordner "folder" werden der Liste hinzugefügt
    for pic in folder:
        image_list.append(mpimage.imread(pic))

    # 1.2 Geben Sie die Liste zurück
    return image_list


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    :param images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    :return D: data matrix that contains the flattened images as rows
    """
    # 2.1 Initalisiere die Datenmatrix mit der richtigen Größe und Typ.
    # Achtung! Welche Dimension hat die Matrix?
    n = len(images)
    x, y = images[0].shape
    data_matrix = np.zeros((n, x*y))


    # 2.2 Fügen Sie die Bilder als Zeilen in die Matrix ein.
    for index, image in enumerate(images):
        data_matrix[index] = image.reshape(x * y)

    # 2.3 Geben Sie die Matrix zurück
    return data_matrix


def calculate_svd(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform SVD analysis for given data matrix.

    :param D: data matrix of size n x m where n is the number of observations and m the number of variables

    :return eigenvec: matrix containing principal components as rows
    :return singular_values: singular values associated with eigenvectors
    :return mean_data: mean that was subtracted from data
    """
    
    # 3.1 Berechnen Sie den Mittelpukt der Daten
    # Tipp: Dies ist in einer Zeile möglich (np.mean, besitzt ein Argument names axis)
    center = np.mean(D, axis=0)
    mean_data = D - center

    # 3.2 Berechnen Sie die Hauptkomponeten sowie die Singulärwerte der ZENTRIERTEN Daten.
    # Dazu können Sie numpy.linalg.svd(..., full_matrices=False) nutzen.
    U, S, V = np.linalg.svd(mean_data, full_matrices=False)

    # 3.3 Geben Sie die Hauptkomponenten, die Singulärwerte sowie den Mittelpunkt der Daten zurück
    return V, S, center

def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    :param singular_values: vector containing singular values
    :param threshold: threshold for determining k (default = 0.8)

    :return k: threshold index
    """

    # 4.1 Normalizieren Sie die Singulärwerte d.h. die Summe aller Singlärwerte soll 1 sein
    singular_values = singular_values / np.sum(singular_values, axis=0, keepdims=1)

    # 4.2 Finden Sie den index k, sodass die ersten k Singulärwerte >= dem Threshold sind.
    k = 0
    added_values = 0

    for value in singular_values:
        # zählt Durchgänge, bis threshold unterschritten wird
        if (added_values >= threshold):
            k = k+1

        added_values = value + added_values
    
    # 4.3 Geben Sie k zurück
    return k


def project_faces(pcs: np.ndarray, mean_data: np.ndarray, images: list) -> np.ndarray:
    """
    Project given image set into basis.

    :param pcs: matrix containing principal components / eigenfunctions as rows
    :param mean_data: mean data that was subtracted before computation of SVD/PCA
    :param images: original input images from which pcs were created

    :return coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """
    # 5.1 Initialisieren Sie die Koeffizienten für die Basis.
    # Sie sollen als Zeilen in einem np.array gespeichert werden.
    a = np.zeros((images.shape[0], pcs.shape[0]))

    # 5.1 Berechnen Sie für jedes Bild die Koeffizienten.
    # Achtung! Denkt daran, dass die Daten zentriert werden müssen.
    for i, image in enumerate(images):

        # center abziehen
        x = image - mean_data
        
        # Koeffizienten mit Skalarprodukt errechnen
        a[i] = np.dot(pcs, x)

    # 5.2 Geben Sie die Koeffizenten zurück
    return a


def identify_faces(coeffs_train: np.ndarray, coeffs_test: np.ndarray) -> (
np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    :param coeffs_train: coefficients for training images, each image is represented in a row
    :param coeffs_test: coefficients for test images, each image is represented in a row

    :return indices: array containing the indices of the found matches
    """
    # Indicee-Array initialisieren
    test_indicees = np.zeros(coeffs_test.shape[0])

    # 8.1 Berechnen Sie für jeden Testvektor den nächsten Trainingsvektor.
    # Achtung! Die Distanzfunktion ist definiert über den Winkel zwischen den Vektoren.
    for i_test, value_test in enumerate(coeffs_test):
        # Array initialisieren für die Abstände vom Test-Koeff. zu jedem Trainings-Koeff. 
        tmp = np.zeros(coeffs_train.shape[0])
        
        for i_train, value_train in enumerate(coeffs_train):
            # Abstand vom Test-Koeff. zum jeweiligen Trainings-Koeff. berechnen
            tmp[i_train] = np.arccos(np.dot(value_test, value_train) / (np.linalg.norm(value_test) * np.linalg.norm(value_train)))

        # Indicee vom kleinsten Abstand ermitteln
        test_indicees[i_test] = np.argmin(tmp)

    return test_indicees   


if __name__ == '__main__':
    ...
    # 1. Aufgabe: Laden Sie die Trainingsbilder.
    # Implementieren Sie dazu die Funktion load_images.
    path = ["data/load_test/1/",
            "data/load_test/2/", 
            "data/test/", 
            "data/train/"]
    
    images = load_images(path[3])

    # 2. Aufgabe: Konvertieren Sie die Bilder zu Vektoren die Sie alle übereinander speichern,
    # sodass sich eine n x m Matrix ergibt (dabei ist n die Anzahl der Bilder und m die Länge des Bildvektors).
    # Implementieren Sie dazu die Funktion setup_data_matrix.
    data_matrix = setup_data_matrix(images)

    # 3. Aufgabe: Finden Sie alle Hauptkomponeten des Datensatztes.
    # Implementieren Sie dazu die Funktion calculate_svd
    V, S, center = calculate_svd(data_matrix)

    # 4. Aufgabe: Entfernen Sie die "unwichtigsten" Basisvektoren.
    # Implementieren Sie dazu die Funktion accumulated_energy um zu wissen wie viele
    # Baisvektoren behalten werden sollen. Plotten Sie Ihr Ergebniss mittels
    # lib.plot_singular_values_and_energy
    k = accumulated_energy(S)

    lib.plot_singular_values_and_energy(S, k)

    # 5. Aufgabe: Projizieren Sie die Trainingsdaten in den gefundenen k-dimensionalen Raum,
    # indem Sie die Koeffizienten für die gefundene Basis finden.
    # Implementieren Sie dazu die Funktion project_faces
    a = project_faces(V, center, data_matrix)

    # 6. Aufgabe: Laden Sie die Test Bilder (load_images).
    images_test = load_images(path[2])
    data_matrix_test = setup_data_matrix(images_test)

    # 7. Aufgabe: Projizieren Sie die Testbilder in den gefundenen k-dimensionalen Raum (project_faces).
    a_test = project_faces(V, center, data_matrix_test)

    # 8. Aufgabe: Berechnen Sie für jedes Testbild das nächste Trainingsbild in dem
    # gefundenen k-dimensionalen Raum. Die Distanzfunktion ist über den Winkel zwischen den Punkten definiert.
    # Implementieren Sie dazu die Funktion identify_faces.
    indicees = identify_faces(a, a_test)

    # Plotten Sie ihr Ergebniss mit der Funktion lib.plot_identified_faces
    lib.plot_identified_faces(indicees.astype(int), images, images_test, V, a_test, center)
    # plot the identified faces
