import numpy as np
import copy

def search(arr, ins_val, low, high):
    fnd_idx = -1
    if (arr.size == 0):
        pass
    else:
        while (low <= high):
            mid = int(low + ((high - low) / 2))
            if (ins_val > (arr[mid])[0]):
                low = mid + 1
                continue
            if (ins_val < (arr[mid])[0]):
                high = mid - 1
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

def __ins(arr, ins_val, index):
    if (arr.size == 0):
        arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
        return arr
    else:
        fnd_idx = search(arr, ins_val, 0, arr.size-1)
        if (fnd_idx >= 0):
            ((arr[fnd_idx])[1])[0] = np.uint32(((arr[fnd_idx])[1])[0]) + np.uint32(1)
            return arr
        else:
            while (index >= 0):
                    if (ins_val > (arr[index - 1])[0]):
                        arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
                        break
                    if (ins_val < (arr[index - 1])[0]):
                        if (index == 0):
                            arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
                        index = index - 1
                        continue
                    else:
                        ((arr[index - 1])[1])[0] = np.uint32(((arr[index - 1])[1])[0]) + np.uint32(1)
                        break
            return arr

def getArrayOfGrayLevelsWithFreq(gray_img, lvldtype=np.uint8):
    aryoflst = np.empty(0, np.dtype([('glvl', lvldtype), ('freq', np.uint32, (1,))]), 'C')
    for x in range(0, (gray_img.shape)[0], 1):
        for y in range(0, (gray_img.shape)[1], 1):
            aryoflst = __ins(aryoflst, gray_img[x, y], index=aryoflst.size)
    return aryoflst
