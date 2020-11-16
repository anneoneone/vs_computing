import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from scipy import fftpack


def show_image(img):
    """
    Shows an image (img) using matplotlib
    """
    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3 or img.shape[-1] == 4:
            plt.imshow(img[...,:3])
        if img.shape[-1] == 1 or img.shape[-1] > 4:
            plt.imshow(img[:,:], cmap="gray")
        plt.show()


def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    # 1.1.1 TODO. Initialisieren Sie das resultierende Bild
    # Codebeispiel: new_img = np.zeros(img.shape)
    new_img = np.zeros(img.shape)                       # in new_img wird das neue Bild geschrieben
    kernel_shape = int(kernel.shape[0] / 2)             # gibt an um wie viel das Bild vergrößert werden muss
    img_edge = np.pad(img, kernel_shape, mode="edge")   # zum rechnen; wird an den rändern vergrößert

    # vergrößertes bild der FFT unterziehen
    img_fft = fftpack.fft2(img_edge)

    # 1.1.2 TODO. Implementieren Sie die Faltung.
    # Achtung: die Faltung (convolution) soll mit beliebig großen Kernels funktionieren.
    # Tipp: Nutzen Sie so gut es geht Numpy, sonst dauert der Algorithmus zu lange.
    # D.h. Iterieren Sie nicht über den Kernel, nur über das Bild. Der Rest geht mit Numpy.

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # temporärer bildausschnitt wird generiert 
            tmp_matrix = img_edge[i : i+kernel.shape[0] , j : j+kernel.shape[0]]

            # multiplikation mit faltung
            mult_mat = np.multiply(tmp_matrix, kernel)
            add_mat = np.sum(mult_mat)

            new_img[i][j] = add_mat

    # Achtung! Achteten Sie darauf, dass wir ein Randproblem haben. Wie ist die Faltung am Rand definiert?
    # Tipp: Es gibt eine Funktion np.pad(Matrix, 5, mode="edge") die ein Array an den Rändern erweitert.
    # 1.1.3 TODO. Returnen Sie das resultierende "Bild"/Matrix

    return new_img


def magnitude_of_gradients(RGB_img):
    """
    Computes the magnitude of gradients using x-sobel and y-sobel 2Dconvolution

    :param img: RGB image
    :return: length of the gradient
    """
    # 3.1.1 TODO. Wandeln Sie das RGB Bild in ein grayscale Bild um.
    img_gray = RGB_img[...,:3]@np.array([0.299, 0.587, 0.114])

    # 3.1.2 TODO: Definieren Sie den x-Sobel Kernel und y-Sobel Kernel.
    kernel_x = np.matrix(
            [[1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]])
    
    kernel_y = np.matrix(
            [ [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]])
    
    # 3.1.3 TODO: Nutzen Sie sie convolution2D Funktion um die Gradienten in x- und y-Richtung zu berechnen.
    g_x = convolution2D(img_gray, kernel_x)
    g_y = convolution2D(img_gray, kernel_y)

    # 3.1.4 TODO: Nutzen Sie die zwei resultierenden Gradienten um die gesammt Gradientenlängen an jedem Pixel auszurechnen.
    g = np.sqrt(pow(g_x, 2) + pow(g_y, 2))   

    return g


# Diese if Abfrage (if __name__ == '__main__':) sorgt dafür, dass der Code nur
# ausgeführt wird, wenn die Datei (mog.py) per python/jupyter ausgeführt wird ($ python mog.py).
# Solltet Ihr die Datei per "import mog" in einem anderen Script einbinden, wird dieser Code übersprungen.
if __name__ == '__main__':
    # Bild laden und zu float konvertieren
    img = mpimage.imread('bilder/tower.jpg')
    img = img.astype("float64")

    # Wandelt RGB Bild in ein grayscale Bild um
    img_gray = img[...,:3]@np.array([0.299, 0.587, 0.114])
    

    # Aufgabe 1.
    # 1.1 TODO: Implementieren Sie die convolution2D Funktion (oben)
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[1][1] = 1

    new_img = convolution2D(img_gray, kernel)

    # Aufgabe 2.
    # 2.1 TODO: Definieren Sie mindestens 5 verschiedene Kernels (darunter sollten beide Sobel sein) und testen Sie sie auf dem grayscale Bild indem Sie convolution2D aufrufen.
    # 2.2 TODO: Speichern Sie alle Resultate als Bilder (sehe Tipp 2). Es sollten 5 Bilder sein.

    # sharpen
    kernel = np.matrix(
            [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]])
    new_img = convolution2D(img_gray, kernel)
    show_image(new_img)
    mpimage.imsave("bilder/a1/sharpen.png", new_img, cmap="gray")
    
    # right_sobel
    kernel = np.matrix(
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])
    new_img = convolution2D(img_gray, kernel)
    show_image(new_img)
    mpimage.imsave("bilder/a1/right_sobel.png", new_img, cmap="gray")

    # bottom_sobel
    kernel = np.matrix(
            [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]])
    new_img = convolution2D(img_gray, kernel)
    show_image(new_img)
    mpimage.imsave("bilder/a1/bottom_sobel.png", new_img, cmap="gray")

    # outline
    kernel = np.matrix(
            [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]])
    new_img = convolution2D(img_gray, kernel)
    show_image(new_img)
    mpimage.imsave("bilder/a1/outline.png", new_img, cmap="gray")

    # emboss
    kernel = np.matrix(
            [[-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]])
    new_img = convolution2D(img_gray, kernel)
    show_image(new_img)
    mpimage.imsave("bilder/a1/emboss.png", new_img, cmap="gray")


    # Aufgabe 3:
    # 3.1 TODO: Implementieren Sie die magnitude_of_gradients Funktion (oben) und testen Sie sie mit dem RGB Bild.
    g = magnitude_of_gradients(img)
    show_image(g)
    
    # 3.2 TODO: Speichern Sie das Resultat als Bild (sehe Tipp 2).
    mpimage.imsave("bilder/gradients.png", g, cmap="gray")

    # ------------------------------------------------
    # Nützliche Funktionen:
    # ------------------------------------------------
    # Tipp 1: So können Sie eine Matrix als Bild anzeigen:
    # show_image(gray)

    # Tipp 2: So können Sie eine NxMx3 Matrix als Bild speichern:
    # mpimage.imsave("test.png", img)
    # und so können Sie eine NxM Matrix als grayscale Bild speichern:
    # mpimage.imsave("test.png", gray, cmap="gray")
