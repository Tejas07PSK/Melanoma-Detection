import numpy as np
import math
from util import Util as u

class HarFeat(object):
    def __init__(self, img, glvlwthfreq):
        self.__glcm = self.__generateGLCM(img, glvlwthfreq)
        (self.__energy, self.__entropy, self.__contrast, self.__idm_homogeneity, self.__dm) = self.__generateHaralickFeatures(self.__glcm, glvlwthfreq)

    def __generateGLCM(self, img, glvlwthfreq):
        coocurmat = np.zeros((glvlwthfreq.size, glvlwthfreq.size), np.uint32, 'C')
        for x in img:
            first = u.search(glvlwthfreq, x[0], 0, glvlwthfreq.size)
            for i in range(1, x.size, 1):
                second = u.search(glvlwthfreq, x[i], 0, glvlwthfreq.size)
                coocurmat[first,second] = np.uint32(coocurmat[first,second]) + np.uint32(1)
                first = second
        return coocurmat

    def __generateHaralickFeatures(self, glcm, glvlwthfreq):
        i = 0
        energy = 0.0
        entropy = 0.0
        contrast = 0.0
        idm_homogeneity = 0.0
        dm = 0.0
        for x in glcm:
            j=0
            for y in x:
                if (y == 0):
                    pass
                else:
                    energy = energy + math.pow(float(y), 2)
                    entropy = entropy + (float(y) * (- math.log(float(y))))
                    contrast = contrast + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2) * float(y))
                    idm_homogeneity = idm_homogeneity + ((1 / (1 + math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))) * float(y))
                    dm = dm + (math.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * float(y))
                j = j + 1
            i = i + 1
        energy = math.sqrt(energy)
        return (energy, entropy, contrast, idm_homogeneity, dm)


    def getGLCM(self):
        return self.__glcm

    def getEnergy(self):
        return self.__energy

    def getEntropy(self):
        return self.__entropy

    def getContrast(self):
        return self.__contrast

    def getHomogeneity(self):
        return self.__idm_homogeneity

    def getDm(self):
        return self.__dm
