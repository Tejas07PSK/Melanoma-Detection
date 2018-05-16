import numpy as np
from util import Util as u

class FeatExt(object):
    def __init__(self, img, glvlwthfreq):
        self.__glcm = self.__generateGLCM(img, glvlwthfreq)

    def __generateGLCM(self, img, glvlwthfreq):
        coocurmat = np.zeros((glvlwthfreq.size, glvlwthfreq.size), np.uint32, 'C')
        for x in img:
            first = u.search(glvlwthfreq, x[0], 0, glvlwthfreq.size)
            for i in range(1, x.size, 1):
                second = u.search(glvlwthfreq, x[i], 0, glvlwthfreq.size)
                coocurmat[first,second] = np.uint32(coocurmat[first,second]) + np.uint32(1)
                first = second
        return coocurmat

    def getGLCM(self):
        return (self.__glcm)