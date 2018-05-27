import numpy as np
import math
from util import Util as u

class HarFeat(object):
    def __init__(self, img, glvlwthfreq, offset=()):
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
                    print(img[i, j])
                    print(img[(i + offset[0]),(j + offset[1])])
                    first = u.search(glvlwthfreq, img[i,j], 0, glvlwthfreq.size-1)
                    second = u.search(glvlwthfreq, img[(i + offset[0]),(j + offset[1])], 0, glvlwthfreq.size-1)
                    print((first,second))
                    coocurmat[first, second] = np.uint32(coocurmat[first, second]) + np.uint32(1)
                    print(coocurmat[first, second])
        return coocurmat

    def generateResizedGLCM(self, src_img, sd, offset=(0,1)):
        coocurmat = np.zeros((sd[2], sd[2]), np.uint32, 'C')
        for i in range(0, (src_img.shape)[0], 1):
            for j in range(0, (src_img.shape)[1], 1):
                if (src_img[i,j] >= sd[2]):
                    continue
                else:
                    if ((((i + offset[0]) < 0) | ((i + offset[0]) >= src_img.shape[0])) | (((j + offset[1]) < 0) | ((j + offset[1]) >= src_img.shape[1]))):
                        continue
                    else:
                        coocurmat[src_img[i,j], src_img[(i + offset[0]),(j + offset[1])]] = np.uint32(coocurmat[src_img[i,j], src_img[(i + offset[0]),(j + offset[1])]]) + np.uint32(1)
        return coocurmat

    def __generateHaralickFeatures(self, glcm, glvlwthfreq):
        i = 0
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
        for x in glcm:
            j=0
            for y in x:
                if (y == 0):
                    pass
                else:
                    asm = asm + math.pow(float(y), 2)
                    entropy = entropy + (float(y) * (- math.log(float(y))))
                    contrast = contrast + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2) * float(y))
                    idm_homogeneity = idm_homogeneity + ((1 / (1 + math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))) * float(y))
                    dm = dm + (math.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * float(y))
                    ux = ux + (float((glvlwthfreq[i])[0]) * float(y))
                    uy = uy + (float((glvlwthfreq[j])[0]) * float(y))
                    m1 = m1 + ((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * float(y))
                    m3 = m3 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 3) * float(y))
                    m4 = m4 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 4) * float(y))
                j = j + 1
            i = i + 1
        (energy, m2) = (math.sqrt(asm), contrast)
        i = 0
        for x in glcm:
            j=0
            for y in x:
                if (y == 0):
                    pass
                else:
                    vx = vx + (math.pow((float((glvlwthfreq[i])[0]) - ux), 2) * float(y))
                    vy = vy + (math.pow((float((glvlwthfreq[j])[0]) - uy), 2) * float(y))
                    correlation = correlation + ((float((glvlwthfreq[i])[0]) - ux) * (float((glvlwthfreq[j])[0]) - uy) * float(y))
                    cluster_shade = cluster_shade + (math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)), 3) * float(y))
                    cluster_prominence = cluster_prominence + (math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)), 4) * float(y))
                    har_correlation = har_correlation + ((float((glvlwthfreq[i])[0]) * float((glvlwthfreq[j])[0]) * float(y)) - math.pow(((ux + uy) / 2), 2))
                j = j + 1
            i = i + 1
        (vx, vy) = (math.sqrt(vx), math.sqrt(vy))
        (correlation, har_correlation) = ((correlation / (vx * vy)), (har_correlation  / math.pow(((vx + vy) / 2), 2)))
        dasm = 0.0
        dmean = 0.0
        dentropy = 0.0
        for k in range(0, glvlwthfreq.size, 1):
            psum = 0.0
            for i in range(0,(glcm.shape)[0], 1):
                 for j in range(0, (glcm.shape)[0], 1):
                        if (math.fabs(i - j) == k):
                            psum = psum + float(glcm[i,j])
                        else:
                            continue
            (dasm, dmean) = ((dasm + math.pow(psum, 2)), (dmean + (k * psum)))
            if (psum <= 0):
                dentropy = dentropy + 0.0
                continue
            else:
                dentropy = dentropy + (psum * (- math.log(psum)))
        return (asm, energy, entropy, contrast, idm_homogeneity, dm, correlation, har_correlation, cluster_shade, cluster_prominence, m1, m2, m3, m4, dasm, dmean, dentropy)

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
    


