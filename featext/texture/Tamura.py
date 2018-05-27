import numpy as np
from util import Util as u
import cv2

class TamFeat(object):

    def __init__(self, img):
        self.__coarseness = self.__generateCoarseness(img)

    def __generateCoarseness(self, src_img):
        sbest = np.zeros(src_img.shape, np.uint32, 'C')
        for x in range(0, (src_img.shape)[0], 1):
            for y in range(0, (src_img.shape)[1], 1):
                emax = np.empty(0, np.dtype([('E', float), ('K', int)]), 'C')
                for k in range(1, 7, 1):
                    emax = np.insert(emax, emax.size, (np.abs(self.__nebAvg(x + np.float_power(2, k-1), y, k, src_img) - self.__nebAvg(x - np.float_power(2, k-1), y, k, src_img)), k-1), 0)
                    emax = np.insert(emax, emax.size, (np.abs(self.__nebAvg(x, y + np.float_power(2, k-1), k, src_img) - self.__nebAvg(x, y - np.float_power(2, k-1), k, src_img)), k-1), 0)
                emax.sort(axis=0, kind='mergesort', order='E')
                sbest[x, y] = np.float_power(2, (emax[emax.size-1])[1])
        return (float(np.sum(sbest, axis=None, dtype=float) / float(sbest.size)))

    def __nebAvg(self, x, y, k, src_img):
        avg = 0.0
        const = np.float_power(2, k-1)
        xh = int(np.round(x + const - 1))
        xl = int(np.round(x - const))
        yh = int(np.round(y + const - 1))
        yl = int(np.round(y - const))
        (xl, xh, yl, yh) = self.__checkSigns(xl, xh, yl, yh, src_img.shape)
        for r in range(xl, xh, 1):
            for c in range(yl, yh, 1):
                avg = avg + (float(src_img[r, c]) / float(np.float_power(2, 2*k)))
        return avg

    def __checkSigns(self, xl, xh, yl, yh, shape):
        if (xl < 0):
            xl = 0
        if (xl > shape[0]):
            xl = shape[0]
        if (xh < 0):
            xh = 0
        if (xh > shape[0]):
            xh = shape[0]
        if (yl < 0):
            yl = 0
        if (yl > shape[1]):
            yl = shape[1]
        if (yh < 0):
            yh = 0
        if (yh > shape[1]):
            yh = shape[1]
        return (xl, xh, yl, yh)

    def __generateContrastAndKurtosis(self, img):
        

    def __generateVariance(self, matrix, matlvls):
        m = np.mean(matrix, axis=None, dtype=float)
        variance = 0.0
        for tup in matlvls:
            sum = sum + (np.float_power((float(tup[0]) - m), 2) * (float((tup[1])[0]) / float(matrix.size)))
        return variance



    def getCoarseness(self):
        return self.__coarseness