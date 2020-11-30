import numpy as np
import matplotlib.pyplot as plt


def distance(a, b):
    """calculates the euclidean distance"""
    return np.linalg.norm(a-b)


def plot_clusters(points, centers, indices):
    # print(indices)

    """plots the clusters"""
    cluster_colors = ['r', 'b', 'y', 'g', 'p', 'c']
    plt.clf()
    plt.axis('equal')
    plt.ion()
    for i in range(len(centers)):
        # Plotting the clusters
        plt.plot(*zip(*points[indices == i]),  marker='.', color=cluster_colors[i], ls='')
        # Plotting cluster centers
        plt.plot(centers[i][0], centers[i][1],  marker=r'$\bowtie$', color='k', markersize=10)
    plt.pause(1)
    plt.show()


def initialize(points, k):
    """
    Initialize k-means by returning k cluster centers. The cluster centers are
    random sample points.

    :param points: ndarray with shape (N,d) containing datasetpoints, where
                   N is number of points and d is the dimension of the points.
    :param k: number of cluster centers.
    :return: ndarray with shape(k,d) containing the random cluster centers
    """
    # 1.1 Initlaizieren Sie das cluster center ndarray
    rows, cols = points.shape
    cluster_center = np.zeros(shape=(k, cols))
    
    # 1.2 Pro cluster center (k) wähle einen zufälligen Punkt als cluster center
    rng = np.random.default_rng()
    random_indices = rng.choice(rows, size=k, replace=False)

    for i in range(0, k):
        cluster_center[i] = points[random_indices[i]]


    # 1.3 Returnen Sie die cluster center
    # print("cluster_center:\n", cluster_center)
    return cluster_center


def assign_to_center(points, centers):
    """
    Assigns each data point to its nearest cluster center (the smallest eucledian distance).
    The function returns the indices and the overall_distance,
    such that the nearest cluster center of point[i] is centers[indices[i]]
    (The overall distance can be used as a measure of global change.)

    :param points: ndarray with shape (N,d) containing datasetpoints, where
                   N is number of points and d is the dimension of the points.
    :param centers: ndarray with shape (k,d) containing the current cluster centers,
                    where k is the number of cluster and d the dimension of the points.
    :return: indices and overall_distance
    """
    # 2.1.1 Berechnen Sie pro Punkt die Distanz zu allen cluster centers und speichern
    # Sie den index des nächsten Clusters sowie den Abstande zu dem cluster centers

    indices = np.zeros(shape=(len(points)))

    overall_distance = 0
    distance_tmp_arr = np.zeros(shape=(len(centers),1))
    
    for i, i_value in enumerate(points):
        for j, j_value in enumerate(centers):
            
            distance_tmp_arr[j] = distance(i_value, j_value)

        indices[i] = np.argmin(distance_tmp_arr)
        overall_distance += np.min(distance_tmp_arr)

    #     print("distance_tmp_arr\n",distance_tmp_arr)
    #     print("distance_tmp_arr\n",np.min(distance_tmp_arr))
    #     print("distance_tmp_arr\n",np.argmin(distance_tmp_arr))

    # print("overall_distance\n", overall_distance)
    # print("indices\n", indices)

    # 2.1.2 Returnen Sie die Indices sowie die Summe der Abstände
    return indices, overall_distance


def update_centers(points, centers, indices):
    """
    Updates the cluster centers depending on the point assignment (indices).
    The new cluster center is the mean of all points belonging to this cluster center.

    :param points: ndarray with shape (N,d) containing datasetpoints, where
                   N is number of points and d is the dimension of the points.
    :param centers: ndarray with shape (k,d) containing the current cluster centers,
                    where k is the number of cluster and d the dimension of the points.
    :param indices: ndarray with shape (N) containing the index of the cluster
                    center for every point.
    :return: new cluster centers
    """
    # init new_centers array
    dimensions = np.shape(points)[1]
    new_centers = np.zeros(shape=(len(centers), dimensions))

    # print("old centers_value\n", centers)

    # 2.2.1 Updaten Sie die cluster centers mit dem Durchschnittspunkt in jedem Cluster
    for centers_index, centers_value in enumerate(centers):
        for points_index, points_value in enumerate(points):
            # take all points of this cluster
            if (indices[points_index] == centers_index):
                
                # add point to array
                centers_value = np.vstack((centers_value, points_value))
                
                print("new centers_value\n", centers_value)

        # delete first entry
        centers_value = np.delete(centers_value, 0, 0)


        print("new centers_value 2\n", centers_value)

        # calc mean of all entries of the array
        new_centers[centers_index] = np.mean(centers_value, axis=0)

    # print("new centers_value\n", new_centers)
    
    return new_centers                
    



def k_means(points, k, iterations=10):
    """
    Assigns each data point to its nearest cluster center (the smallest eucledian distance).
    The function returns the indices and the overall_distance,
    such that the nearest cluster center of point[i] is centers[indices[i]]
    (The overall distance can be used as a measure of global change.)

    :param points: ndarray with shape (N,d) containing datasetpoints, where
                   N is number of points and d is the dimension of the points.
    :param k: number of cluster centers.
    :param iterations: number of k-means iterations.
    :return: cluster centers und indices
    """
    # 1. Initializieren Sie die cluster centers und die indices.
    # Implementieren Sie dazu die Funktion initialize
    cluster_centers = initialize(points, k)

    # 2. Pro iteration:
    for i in range(0, iterations):
        # 2.1 Weisen Sie den Punkten die jeweiligen cluster center zu
        # Implementieren Sie dazu die Funktion assign_to_center
        indices, overall_distance = assign_to_center(points, cluster_centers)

        # 2.2 Aktualisieren Sie die neuen cluster center anhand der berechneten indices.
        # Implementieren Sie dazu die Funktion update_centers
        cluster_centers = update_centers(points, cluster_centers, indices)

        # 2.3 (optional) Plotten Sie die cluster und Datenpunkte mittels plot_clusters
        # plot_clusters(points, cluster_centers, indices)

        # 2.4 (optional) Brechen Sie die Schleife vorzeitig ab sollten sich die
        # Distanz zu den cluster centern kaum noch verändert haben

    # 3. Return cluster centers und indices
    return cluster_centers, indices


# ---------------------------------------------------------------------------
# k-means Beispielaufrufe (Hier unten ist alles funktionstüchtig)
# Lesen Sie sich den Bereich dennoch durch.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initializierung
    num_clusters = 4
    num_iter = 5

    # Laden der Datenpunkte
    points = np.loadtxt('kmeans_points_small.txt')

    # Aufrufen von k-means mit den Datenpunkten
    # TODO: Implementieren Sie k_means:
    centers, indices = k_means(points, num_clusters, num_iter)

    # Plotten des Ergebnisses
    plot_clusters(points, centers, indices)

    # Wenn k-means funktionstüchtig ist, kann dieser Bereich einkommentiert werden
    # Mit diesm Beispielcode kann in einem Bild die Anzahl der Farben reduziert
    # werden (nötig um ein z.B. ein gif Bild zu generieren):
    # img = plt.imread("images/Broadway_tower.jpg").copy()
    # cc, ci = k_means(img.reshape(-1,3), 4, num_iter)
    # for i in range(len(cc)):
    #     map = ci.reshape(img.shape[:2])
    #     img[map == i,:] = cc[i]
    
    # plt.imshow(img)
    # plt.show()
