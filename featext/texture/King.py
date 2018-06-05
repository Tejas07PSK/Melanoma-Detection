import numpy as np
from util import Util as u

class KingFeat(object):

    def __init__(self, img, d=2, e=0.3):
        glvlwthfreq = u.getArrayOfGrayLevelsWithFreq(img)
        self.__ngtdm = self.__generateNGTDM(img, glvlwthfreq, d)
        (self.__coarseness, factor) = self.__generateKingsCoarseness(glvlwthfreq, img.size, e)
        self.__contrast = self.__generateKingsContrast(glvlwthfreq, img.size)
        self.__busyness = self.__generateBusyness(glvlwthfreq, img.size, factor)
        self.__complexity = self.__generateComplexity(glvlwthfreq, img.size)
        self.__strength = self.__generateStrength(glvlwthfreq, img.size, e)

    def __generateNGTDM(self, img, glvlwthfreq, d):
        ngtdm = np.zeros(glvlwthfreq.shape, float, 'C')
        for i in range(0, (img.shape)[0], 1):
            for j in range(0, (img.shape)[1], 1):
                if (img[i, j] == 0):
                    continue
                else:
                    index = u.search(glvlwthfreq, img[i, j], 0, glvlwthfreq.size - 1)
                    ngtdm[index] = ngtdm[index] + np.fabs(img[i, j] - (self.__calculateSubSum(img, i, j, d) / (np.power(((2 * d) + 1), 2) - 1)))
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
        if (x >= shape[0]):
            x = shape[0] - 1
        if (y < 0):
            y = 0
        if (y >= shape[1]):
            y = shape[1] - 1
        return (x, y)

    def __generateKingsCoarseness(self, glvlwthfreq, totpix, e):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            sum = sum + ((float((glvlwthfreq[i])[1]) / float(totpix)) * self.__ngtdm[i])
        return ((1 /(e + sum)), sum)

    def __generateKingsContrast(self, glvlwthfreq, totpix):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
              if((glvlwthfreq[i])[0] == (glvlwthfreq[j])[0]):
                  continue
              else:
                  sum = sum + (((float((glvlwthfreq[i])[1])) / float(totpix)) * ((float((glvlwthfreq[j])[1])) / float(totpix)) * np.power((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))
        sum = sum * (1.0 / float(glvlwthfreq.size * (glvlwthfreq.size - 1))) * ((1.0 / np.power(float(totpix), 2)) * (self.__ngtdm).sum(axis=None, dtype=float))
        return sum

    def __generateBusyness(self, glvlwthfreq, totpix, factor):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                    sum = sum + ((float((glvlwthfreq[i])[0]) * ((float((glvlwthfreq[i])[1])) / float(totpix))) - (float((glvlwthfreq[j])[0]) * ((float((glvlwthfreq[j])[1])) / float(totpix))))
        sum = factor / sum
        return sum

    def __generateComplexity(self, glvlwthfreq, totpix):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                sum = sum + ((np.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) / (np.power(float(totpix), 2) * (((float((glvlwthfreq[i])[1])) / float(totpix)) + ((float((glvlwthfreq[j])[1])) / float(totpix))))) * ((((float((glvlwthfreq[i])[1])) / float(totpix)) * self.__ngtdm[i]) + (((float((glvlwthfreq[j])[1])) / float(totpix)) * self.__ngtdm[j])))
        return sum

    def __generateStrength(self, glvlwthfreq, totpix, e):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                sum = sum + ((((float((glvlwthfreq[i])[1])) / float(totpix)) + ((float((glvlwthfreq[j])[1])) / float(totpix))) * np.power((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))
        sum = sum / (e + (self.__ngtdm).sum(axis=None, dtype=float))
        return sum

    def getNGTDM(self):
        return self.__ngtdm

    def getKingsCoarseness(self):
        return self.__coarseness

    def getKingsContrast(self):
        return self.__contrast

    def getKingsBusyness(self):
        return self.__busyness

    def getKingsComplexity(self):
        return self.__complexity

    def getKingsStrength(self):
        return self.__strength