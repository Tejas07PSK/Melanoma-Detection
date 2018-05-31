import numpy as np
import math
from util import Util as u

class HarFeat(object):

    def __init__(self, img, offset=()):
        glvlwthfreq = u.getArrayOfGrayLevelsWithFreq(img)
        if (len(offset) == 0):
            self.__glcm = self.__generateGLCM(img, glvlwthfreq)
        else:
            self.__glcm = self.__generateGLCM(img, glvlwthfreq, offset)
        (self.__asm, self.__energy, self.__entropy, self.__contrast, self.__idm_homogeneity, self.__dm, self.__correlation, self.__har_correlation, self.__cluster_shade, self.__cluster_prominence, self.__moment1, self.__moment2, self.__moment3, self.__moment4, self.__dasm, self.__dmean, self.__dentropy) = self.__generateHaralickFeatures(self.__glcm, glvlwthfreq)

    def __generateGLCM(self, img, glvlwthfreq, offset=(0,1)):
        coocurmat = np.zeros((glvlwthfreq.size, glvlwthfreq.size), np.uint32, 'C')
        for i in range(0, (img.shape)[0], 1):
            for j in range(0, (img.shape)[1], 1):
                if ((((i + offset[0]) < 0) | ((i + offset[0]) >= img.shape[0])) | (((j + offset[1]) < 0) | ((j + offset[1]) >= img.shape[1]))):
                    continue
                else:
                    first = u.search(glvlwthfreq, img[i,j], 0, glvlwthfreq.size-1)
                    second = u.search(glvlwthfreq, img[(i + offset[0]),(j + offset[1])], 0, glvlwthfreq.size-1)
                    coocurmat[first, second] = np.uint32(coocurmat[first, second]) + np.uint32(1)
        return coocurmat

    def __generateHaralickFeatures(self, glcm, glvlwthfreq):
        sumofglcm = glcm.sum(axis=None, dtype=float)
        asm = 0.0
        correlation = 0.0
        har_correlation = 0.0
        entropy = 0.0
        contrast = 0.0
        idm_homogeneity = 0.0
        cluster_shade = 0.0
        cluster_prominence = 0.0
        m1 = 0.0
        m3 = 0.0
        m4 = 0.0
        dm = 0.0
        ux = 0.0
        uy = 0.0
        vx = 0.0
        vy = 0.0
        (energy, m2, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4) = self.__genHarFeatPt1(glcm, glvlwthfreq, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4, sumofglcm)
        (cluster_shade, cluster_prominence, correlation, har_correlation) = self.__genHarFeatPt2(glcm, glvlwthfreq, ux, uy, vx, vy, correlation, cluster_shade, cluster_prominence, har_correlation, sumofglcm)
        dasm = 0.0
        dmean = 0.0
        dentropy = 0.0
        for k in range(0, glvlwthfreq.size, 1):
            psum = 0.0
            for i in range(0,(glcm.shape)[0], 1):
                 for j in range(0, (glcm.shape)[1], 1):
                        if (math.fabs(i - j) == k):
                            psum = psum + (float(glcm[i,j]) / sumofglcm)
                        else:
                            continue
            (dasm, dmean) = ((dasm + math.pow(psum, 2)), (dmean + (k * psum)))
            if (psum <= 0.0):
                dentropy = dentropy + 0.0
                continue
            else:
                dentropy = dentropy + (psum * (- math.log(psum)))
        return (asm, energy, entropy, contrast, idm_homogeneity, dm, correlation, har_correlation, cluster_shade, cluster_prominence, m1, m2, m3, m4, dasm, dmean, dentropy)

    def __genHarFeatPt1(self, glcm, glvlwthfreq, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4, sumofglcm):
        i = 0
        for x in glcm:
            j=0
            for item in x:
                y = float(item) / sumofglcm
                if (y == 0.0):
                    pass
                else:
                    asm = asm + math.pow(y, 2)
                    entropy = entropy + (y * (- math.log(y)))
                    contrast = contrast + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2) * y)
                    idm_homogeneity = idm_homogeneity + ((1 / (1 + math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))) * y)
                    dm = dm + (math.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * y)
                    ux = ux + (float((glvlwthfreq[i])[0]) * y)
                    uy = uy + (float((glvlwthfreq[j])[0]) * y)
                    m1 = m1 + ((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * y)
                    m3 = m3 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 3) * y)
                    m4 = m4 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 4) * y)
                j = j + 1
            i = i + 1
        return (math.sqrt(asm), contrast, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4)

    def __genHarFeatPt2(self, glcm, glvlwthfreq, ux, uy, vx, vy, correlation, cluster_shade, cluster_prominence, har_correlation, sumofglcm):
        i = 0
        for x in glcm:
            j = 0
            for item in x:
                y = float(item) / sumofglcm
                if (y == 0.0):
                    pass
                else:
                    vx = vx + (math.pow((float((glvlwthfreq[i])[0]) - ux), 2) * y)
                    vy = vy + (math.pow((float((glvlwthfreq[j])[0]) - uy), 2) * y)
                    correlation = correlation + ((float((glvlwthfreq[i])[0]) - ux) * (float((glvlwthfreq[j])[0]) - uy) * y)
                    cluster_shade = cluster_shade + (math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)), 3) * y)
                    cluster_prominence = cluster_prominence + (math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)), 4) * y)
                    har_correlation = har_correlation + ((float((glvlwthfreq[i])[0]) * float((glvlwthfreq[j])[0]) * y) - math.pow(((ux + uy) / 2), 2))
                j = j + 1
            i = i + 1
        (vx, vy) = (math.sqrt(vx), math.sqrt(vy))
        (correlation, har_correlation) = ((correlation / (vx * vy)), (har_correlation / math.pow(((vx + vy) / 2), 2)))
        return (cluster_shade, cluster_prominence, correlation, har_correlation)

    def getGLCM(self):
        return self.__glcm

    def getAngularSecondMomentASM(self):
        return self.__asm

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

    def getCorrelation(self):
        return self.__correlation

    def getHarCorrelation(self):
        return self.__har_correlation

    def getClusterShade(self):
        return self.__cluster_shade

    def getClusterProminence(self):
        return self.__cluster_prominence

    def getMoment1(self):
        return self.__moment1

    def getMoment2(self):
        return self.__moment2

    def getMoment3(self):
        return self.__moment3

    def getMoment4(self):
        return self.__moment4

    def getDasm(self):
        return self.__dasm

    def getDmean(self):
        return self.__dmean

    def getDentropy(self):
        return self.__dentropy
    


