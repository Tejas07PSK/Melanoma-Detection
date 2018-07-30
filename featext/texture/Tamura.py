import numpy as np
from util import Util as u
import cv2
from threading import Thread, Lock, Event
from queue import Queue
import time

class TamFeat(object):

    q = Queue(maxsize=4)

    def __init__(self, img):
        t = time.time()
        (self.__coarseness, varCrs) = self.__generateCoarseness(img)
        print("Coarseness Calc-Time : %f secs\n" % (time.time() - t))
        (self.__contrast, self.__kurtosis, varCon) = self.__generateContrastAndKurtosis(img)
        self.__img_hor_x = cv2.filter2D(img, -1, np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.int16))
        self.__img_vert_y = cv2.filter2D(img, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.int16))
        self.__delg_img = np.round((np.add(self.__img_hor_x, self.__img_vert_y, dtype=float) * 0.5)).astype(np.int8)
        self.__theta_img = np.tanh(np.divide((self.__img_vert_y).astype(float), (self.__img_hor_x).astype(float), dtype=float, out=np.zeros_like((self.__img_vert_y).astype(float)), where=self.__img_hor_x != 0)) + (float(np.pi) / 2.0)
        (self.__linelikeness, varLin) = self.__generateLineLikeness(self.__delg_img, self.__theta_img)
        (self.__directionality, varDir) = self.__generateDirectionality(self.__delg_img, self.__theta_img)
        self.__regularity = self.__generateRegularity(np.sqrt(varCrs), np.sqrt(varDir), np.sqrt(varCon), np.sqrt(varLin))
        self.__roughness = self.__generateRoughness(self.__coarseness, self.__contrast)

    def __generateCoarseness(self, src_img):
        def __tds_opt(tds, mode='s'):
            for t in tds:
                if (mode == 's'):
                    t.start()
                else:
                    t.join()
        lock = Lock()
        sbest = np.zeros(src_img.shape, np.uint32, 'C')
        for x in range(0, (src_img.shape)[0], 1):
            for y in range(0, (src_img.shape)[1], 1):
                emax = np.empty(0, np.dtype([('E', float), ('K', int)]), 'C')
                #print((x,y))
                for k in range(1, 7, 1):
                    tds = [Thread(target=self.__nebAvg, name='Cor0', args=(x + np.float_power(2, k-1), y, k, src_img, lock, Event(), 0)), Thread(target=self.__nebAvg, name='Cor1', args=(x - np.float_power(2, k-1), y, k, src_img, lock, Event(), 1)), Thread(target=self.__nebAvg, name='Cor2', args=(x, y + np.float_power(2, k-1), k, src_img, lock, Event(), 2)), Thread(target=self.__nebAvg, name='Cor3', args=(x, y - np.float_power(2, k-1), k, src_img, lock, Event(), 3))]
                    __tds_opt(tds)
                    __tds_opt(tds, 'j')
                    nbavgs = self.__getFromQueue()
                    emax = np.insert(emax, emax.size, (np.abs(nbavgs[0] - nbavgs[1]), k-1), 0)
                    emax = np.insert(emax, emax.size, (np.abs(nbavgs[2] - nbavgs[3]), k-1), 0)
                    #emax = np.insert(emax, emax.size, (np.abs(self.__nebAvg(x + np.float_power(2, k-1), y, k, src_img) - self.__nebAvg(x - np.float_power(2, k-1), y, k, src_img)), k-1), 0)
                    #emax = np.insert(emax, emax.size, (np.abs(self.__nebAvg(x, y + np.float_power(2, k-1), k, src_img) - self.__nebAvg(x, y - np.float_power(2, k-1), k, src_img)), k-1), 0)
                emax.sort(axis=0, kind='mergesort', order='E')
                sbest[x, y] = np.float_power(2, (emax[emax.size-1])[1])
        varCrs = self.__generateVariance(u.getArrayOfGrayLevelsWithFreq(sbest, lvldtype=np.uint32), np.mean(sbest, axis=None, dtype=float))
        return ((float(np.sum(sbest, axis=None, dtype=float) / float(sbest.size))), varCrs)

    def __nebAvg(self, x, y, k, src_img, lck, evt, pos):
        lck.acquire()
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
        (TamFeat.q).put((avg, pos))
        lck.release()
        evt.set()
        #return avg

    def __getFromQueue(self):
        nbavgs = [0.0, 0.0, 0.0, 0.0]
        while ((TamFeat.q).empty() == False):
            item = (TamFeat.q).get()
            nbavgs[ item[1] ] = item[0]
            (TamFeat.q).task_done()
        (TamFeat.q).join()
        return nbavgs

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

    def __generateContrastAndKurtosis(self, src_img):
        glvlwthfreq = u.getArrayOfGrayLevelsWithFreq(src_img)
        m = np.mean(src_img, axis=None, dtype=float)
        variance = self.__generateVariance(glvlwthfreq, m)
        kurtosis = 0.0
        for tup in glvlwthfreq:
            kurtosis = kurtosis + (np.float_power((float(tup[0]) - m), 4) * (float(tup[1]) / float(src_img.size)))
        kurtosis = kurtosis / np.float_power(variance, 2)
        contrast = float(np.sqrt(variance)) / np.float_power(kurtosis, 0.25)
        return (contrast, kurtosis, variance)

    def __generateVariance(self, matlvls, m):
        gls = np.ascontiguousarray(matlvls['glvl'], dtype=float)
        frq = np.ascontiguousarray(matlvls['freq'], dtype=float)
        totpix = frq.sum(axis=None, dtype=float)
        variance = 0.0
        for g in range(0, matlvls.size, 1):
            variance = variance + (np.float_power((gls[g] - m), 2) * (frq[g] / totpix))
        return variance

    def __generateLineLikeness(self, delg_img, theta_img, d=4, t=12):
        dirlevels = u.getArrayOfGrayLevelsWithFreq(theta_img, lvldtype=float)
        ditfctcm = np.zeros((dirlevels.size, dirlevels.size), dtype=np.uint32, order='C')
        for i in range(0, (theta_img.shape)[0], 1):
            for j in range(0, (theta_img.shape)[1], 1):
                if (np.fabs(delg_img[i,j]) > t):
                    x = int(np.round(np.fabs(d * np.cos(theta_img[i, j]))))
                    y = int(np.round(np.fabs(d * np.sin(theta_img[i, j]))))
                    if ((x < 0) | (x >= (theta_img.shape)[0]) | (y < 0) | (y >= (theta_img.shape)[1])):
                        continue
                    else:
                        if ((theta_img[x, y] > (theta_img[i, j] - 1)) & (theta_img[x, y] < (theta_img[i, j] + 1))):
                             idx1, idx2 = u.search(dirlevels, theta_img[i, j], 0, dirlevels.size-1), u.search(dirlevels, theta_img[x, y], 0, dirlevels.size-1)
                             ditfctcm[idx1, idx2] = ditfctcm[idx1, idx2] + 1
                        else:
                            continue
        varLin = self.__generateVariance(u.getArrayOfGrayLevelsWithFreq(ditfctcm, lvldtype=np.uint32), np.mean(ditfctcm, axis=None, dtype=float))
        return (self.__lineLikenessSubPart(ditfctcm, dirlevels), varLin)

    def __lineLikenessSubPart(self, ditfctcm, dirlevels):
        dir = 0.0
        for i in range(0, (ditfctcm.shape)[0], 1):
            for j in range(0, (ditfctcm.shape)[0], 1):
                dir = dir + float(ditfctcm[i, j]) * np.cos((((dirlevels[i])[0] - (dirlevels[j])[0]) * 2.0 * np.pi) / dirlevels.size)
        dir = dir / ditfctcm.sum(axis=None, dtype=float)
        return dir

    def __generateDirectionality(self, delg_img, theta_img, t=12):
        temp = np.zeros_like(theta_img)
        for i in range(0, (delg_img.shape)[0], 1):
            for j in range(0, (delg_img.shape)[1], 1):
                if (delg_img[i, j] > t):
                    temp[i, j] = theta_img[i, j]
        varDir = self.__generateVariance(u.getArrayOfGrayLevelsWithFreq(temp, lvldtype=float), np.mean(temp, axis=None, dtype=float))
        return ((1 / np.sqrt(varDir)), varDir)

    def __generateRegularity(self, sdCrs, sdDir, sdCon, sdLin, r=0.4):
        return  (1 - (r * (sdCrs + sdDir + sdCon + sdLin)))

    def __generateRoughness(self, coarseness, contrast):
        return (contrast + coarseness)

    def getCoarseness(self):
        return self.__coarseness

    def getContrast(self):
        return self.__contrast

    def getKurtosis(self):
        return self.__kurtosis

    def getPrewittHorizontalEdgeImg(self):
        return self.__img_hor_x

    def getPrewittVerticalEdgeImg(self):
        return self.__img_vert_y

    def getCombinedPrewittImg(self):
        return (self.__delg_img).astype(np.uint8)

    def getPrewittDirFactOfImg(self):
        return self.__theta_img

    def getLineLikeness(self):
        return self.__linelikeness

    def getDirectionality(self):
        return self.__directionality

    def getRegularity(self):
        return self.__regularity

    def getRoughness(self):
        return self.__roughness