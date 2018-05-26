import numpy as np
import copy

def search(arr,ins_val,l,h):
    fnd_idx = -1
    if (arr.size == 0):
        return fnd_idx
    else:
        if (l >= h):
            return fnd_idx
        else:
            mid = np.uint8((l+h)/2)
            if (ins_val > (arr[mid])[0]):
                fnd_idx = search(arr,ins_val,l=mid+1,h=h)
            elif (ins_val < (arr[mid])[0]):
                fnd_idx = search(arr,ins_val,l,h=mid-1)
            elif (ins_val == (arr[mid])[0]):
                fnd_idx = mid
            else:
                pass
    return fnd_idx

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


def partition(arr, low, high):
    i = (low - 1)
    pivot = (arr[high])[0]
    for j in range(low, high,1):
        if (arr[j])[0] <= pivot:
            i = i + 1
            temp = copy.deepcopy(arr[i])
            arr[i] = copy.deepcopy(arr[j])
            arr[j] = copy.deepcopy(temp)
    temp2 = copy.deepcopy(arr[i+1])
    arr[i+1] = copy.deepcopy(arr[high])
    arr[high] = copy.deepcopy(temp2)
    return (i+1)
