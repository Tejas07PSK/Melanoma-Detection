import numpy as np

def search(arr,ins_val,l,h):
    fnd_idx = -1
    if (arr.size == 0):
        return (fnd_idx)
    else:
        if (l >= h):
            return (fnd_idx)
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
    return (fnd_idx)