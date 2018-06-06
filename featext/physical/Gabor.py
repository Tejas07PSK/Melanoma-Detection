import numpy as np
import cv2

class Gabor(object):

        def __init__(self, img, corr_colimg, imtype='color'):
            tup = cv2.findContours(cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.__gblurimg = tup[0]
            self.__contours = tup[1]
            self.__hierarchy = tup[2]
            cnt = self.__selectMostAptContourIndex()
            self.__momLstForConts = self.__generateMoments()
            self.__centroidLstForConts = self.__generateCentroidOfCnts()
            (self.__arLstForConts, self.__periLstForConts) = self.__generateAreaNPeriOfCnts()
            self.__selecCntImg = self.__generateContourImg(imtype, cnt)
            (self.__imgcovrect, self.__meanEdge) = self.__generateBoundingRectRotated(imtype, cnt)
            (self.__imgcovcirc, self.__rad) = self.__generateMinEncCirc(imtype, cnt)
            self.__asyidxofles = self.__generateAsymmetryIndex(totar=img.size, cnt=cnt)
            self.__cmptidx = self.__generateCompactIndex(cnt)
            self.__fracdimen = self.__generateFractalDimension()
            self.__diameter = self.__calculateDiameter()
            self.__colorvar = self.__generateColorVariance(corr_colimg)

        def __selectMostAptContourIndex(self):
            tmplst = [ len(c) for c in self.__contours ]
            return tmplst.index(max(tmplst))

        def __generateMoments(self):
            moments = []
            for c in self.__contours:
                moments.append(cv2.moments(c))
            return moments

        def __generateCentroidOfCnts(self):
            centroids = []
            for mlc in self.__momLstForConts:
                coorX = int(mlc['m10'] / mlc['m00'])
                coorY = int(mlc['m01'] / mlc['m00'])
                centroids.append((coorX, coorY))
            return centroids

        def __generateAreaNPeriOfCnts(self):
            areas = []
            peri = []
            for c in self.__contours:
                areas.append(cv2.contourArea(c))
                peri.append(cv2.arcLength(c, True))
            return (areas, peri)

        def __generateContourImg(self, imtype='gray', cnt=0):
            if (imtype == 'gray'):
                return cv2.drawContours((self.__gblurimg).copy(), [self.__contours[cnt]], 0, 255, 2, cv2.LINE_AA)
            else:
                tmp = cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR)
                return cv2.drawContours(tmp, [self.__contours[cnt]], 0, (0,255,0), 2, cv2.LINE_AA)

        def __generateBoundingRectRotated(self, imtype='gray', cnt=0):
            rect = cv2.minAreaRect(self.__contours[cnt])
            mean_edge = ((rect[1])[0] + ((rect[1])[0])) / 2
            if (imtype == 'gray'):
                return (cv2.drawContours((self.__gblurimg).copy(), [np.int0(cv2.boxPoints(rect))], 0, 255, 2, cv2.LINE_AA), mean_edge)
            else:
                return (cv2.drawContours(cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR), [np.int0(cv2.boxPoints(rect))], 0, (0,255,0), 2, cv2.LINE_AA), mean_edge)

        def __generateMinEncCirc(self, imtype='gray', cnt=0):
            (x, y), radius = cv2.minEnclosingCircle(self.__contours[cnt])
            center = (int(x), int(y))
            radius = int(radius)
            if (imtype == 'gray'):
                return (cv2.circle((self.__gblurimg).copy(), center, radius, 255, 2, cv2.LINE_AA), radius)
            else:
                return (cv2.circle(cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR), center, radius, (0,255,0), 2, cv2.LINE_AA), radius)

        def __generateAsymmetryIndex(self, totar, cnt=0):
            return ((self.__arLstForConts[cnt] / totar) * 100)

        def __generateCompactIndex(self, cnt=0):
            return (np.power(self.__periLstForConts[cnt], 2) / (4 * np.pi * self.__arLstForConts[cnt]))

        def __generateFractalDimension(self):
            return (np.log(self.__meanEdge) / np.log(1 / self.__meanEdge))

        def __calculateDiameter(self):
            return (2 * self.__rad)

        def __generateColorVariance(self, colimg):
            return (np.var(cv2.cvtColor(colimg, cv2.COLOR_BGR2HSV), axis=None, dtype=float))

        def getGaussianBlurredImage(self):
            return self.__gblurimg

        def getListOfContourPoints(self):
            return self.__contours

        def getHierarchyOfContours(self):
            return self.__hierarchy

        def getListOfMomentsForCorrespondingContours(self):
            return self.__momLstForConts

        def getListOfCentroidsForCorrespondingContours(self):
            return self.__centroidLstForConts

        def getListOfAreasForCorrespondingContours(self):
            return self.__arLstForConts

        def getListOfPerimetersForCorrespondingContours(self):
            return self.__periLstForConts

        def getSelectedContourImg(self):
            return self.__selecCntImg

        def getBoundingRectImg(self):
            return self.__imgcovrect

        def getMeanEdgeOfCoveringRect(self):
            return self.__meanEdge

        def getBoundedCircImg(self):
            return self.__imgcovcirc

        def getBoundedCircRadius(self):
            return self.__rad

        def getAsymmetryIndex(self):
            return self.__asyidxofles

        def getCompactIndex(self):
            return self.__cmptidx

        def getFractalDimension(self):
            return self.__fracdimen

        def getDiameter(self):
            return self.__diameter

        def getColorVariance(self):
            return self.__colorvar




