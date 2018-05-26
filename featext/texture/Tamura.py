import numpy as np
from util import Util as u
import cv2

class TamFeat(object):
    def __init__(self, img):
        self.coarseness = self.__generateCoarseness(img)

    def __generateCoarseness(self, src_img):
        sbest = np.zeros(src_img.shape, np.unit32, 'C')
        emax = np.zeros(12, np.dtype([('E', float, (1,)), ('K', int, (1,))]), 'C')
        for x in range(0, (src_img.shape)[0], 1):
            for y in range(0, (src_img.shape)[0], 1):
                count = 0
                for k in range(0, 6, 1):
                    



    def __nebAvg(self, x, y, k, src_img):
        avg = 0.0
        const = np.pow(2, k-1)
        xh = int(np.round(np.abs(x + const - 1)))
        xl = int(np.round(np.abs(x - const)))
        yh = int(np.round(np.abs(y + const - 1)))
        yl = int(np.round(np.abs(y - const)))
        if ((((xh < 0) | (xh > (src_img.shape)[0])) | ((xl < 0) | (xl >= (src_img.shape)[0]))) | (((yh < 0) | (yh > (src_img.shape)[1]) | ((yl < 0) | (yl >= (src_img.shape)[1]))))):
            pass
        else:
            for r in range(xl, xh, 1):
                for c in range(yl, yh, 1):
                    avg = avg + (float(src_img[r, c]) / float(np.pow(2, 2*k)))
        return avg
