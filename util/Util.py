import numpy as np
import copy

def search(arr,ins_val,l,h):
    fnd_idx = -1
    if (arr.size == 0):
        pass
    else:
        while (l < h):
            print("in while (%d,%d) \n" % (l,h))
            mid = np.uint8((l + h) / 2)
            if (ins_val > (arr[mid])[0]):
                l = mid + 1
                continue
            if (ins_val < (arr[mid])[0]):
                h = mid - 1
                continue
            if (ins_val == (arr[mid])[0]):
                fnd_idx = mid
                break
    return fnd_idx

def quickSort(arr, low, high):
    if low < high:
        pi = __partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


def __partition(arr, low, high):
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

def __ins(arr, ins_val, index, isSearched):
    if (arr.size == 0):
        arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
    else:
        flag = 0
        if (isSearched == 0):
            fnd_idx = search(arr, ins_val, 0, arr.size)
            print(ins_val)
            print (fnd_idx)
            if (fnd_idx >= 0):
                flag = 1
                ((arr[fnd_idx])[1])[0] = np.uint32(((arr[fnd_idx])[1])[0]) + np.uint32(1)
        if (flag == 1):
            pass
        elif (ins_val > (arr[index - 1])[0]):
             arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
        elif (ins_val < (arr[index - 1])[0]):
             if (index == 0):
                arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
             else:
                arr = __ins(arr, ins_val, index=index - 1, isSearched=1)
        else:
            ((arr[index - 1])[1])[0] = np.uint32(((arr[index - 1])[1])[0]) + np.uint32(1)
    return arr

def getArrayOfGrayLevelsWithFreq(gray_img, lvldtype=np.uint8):
    aryoflst = np.empty(0, np.dtype([('glvl', lvldtype), ('freq', np.uint32, (1,))]), 'C')
    for x in range(0, (gray_img.shape)[0], 1):
        for y in range(0, (gray_img.shape)[1], 1):
            aryoflst = __ins(aryoflst, gray_img[x, y], index=aryoflst.size, isSearched=0)
    return aryoflst
