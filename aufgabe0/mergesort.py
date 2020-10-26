import numpy as np
import math

def mergesort(list) -> (list):

    if len(list) <= 1:
        return list
    else:
        leftSide, rightSide = split_list(list)

        leftSide = mergesort(leftSide)
        rightSide = mergesort(rightSide)

        mergedlist = mergelist(leftSide, rightSide)

        return mergedlist

    # pass does not do anything but is necessary to run this code
    pass

def split_list(a_list):
    half = len(a_list)//2

    return a_list[:half], a_list[half:]

def mergelist(leftSide, rightSide):
    i = 0
    il = 0
    y = []

    nl = len(leftSide)-1
    nr = len(rightSide)-1

    for i in range(0, sum([nl, nr], 2)):
        if il > nl:
            y.append(rightSide[i-il])
        elif il < i-nr:
            y.append(leftSide[il])
            il = il+1
        elif leftSide[il] <= rightSide[i-il]:
            y.append(leftSide[il])
            il = il+1
        else:
            y.append(rightSide[i-il])

    return y

def printArray(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()


if __name__ == '__main__':
    # use np.arange to create an numpy array and
    # shuffle all values using np.random.shuffle
    # please read the document for these functions
    unsortedList = np.arange(15)
    np.random.shuffle(unsortedList)

    printArray(unsortedList)

    sortedList = mergesort(unsortedList)

    printArray(sortedList)