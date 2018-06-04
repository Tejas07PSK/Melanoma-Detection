import numpy as np
from util import Util as u

class HarFeat(object):

    def __init__(self, img, d=2):
        glvlwthfreq = u.getArrayOfGrayLevelsWithFreq(img)
        self.__ngtdm = self.__generateNGTDM(img, glvlwthfreq, d)

    def __generateNGTDM(self, img, glvlwthfreq, d):
        ngtdm = np.zeros(glvlwthfreq.shape, float, 'C')
        for i in range(0, (img.shape)[0], 1):
            for j in range(0, (img.shape)[1], 1):
                if (img[i, j] == 0):
                    continue
                else:
                    index = u.search(glvlwthfreq, img[i, j], 0, glvlwthfreq.size - 1)
                    ngtdm[index] = ngtdm[index] + (img[i, j] - (self.__calculateSubSum(img, i, j, d) / (np.power(((2 * d) - 1), 2) - 1)))
        return ngtdm

    def __calculateSubSum(self, img, i, j, d):
        sum = 0.0
        m = -d
        while(m < d):
            n = -d
            while(n < d):
                (x, y) = self.__checkLimits((i + m), (j + n), img.shape)
                sum = sum + img[x, y]
                n = n + 1
            m = m + 1
        return sum

    def __checkLimits(self, x, y, shape):
        if (x < 0):
            x = 0
        if (x > shape[0]):
            x = shape[0]
        if (y < 0):
            y = 0
        if (y > shape[1]):
            y = shape[1]
        return (x, y)