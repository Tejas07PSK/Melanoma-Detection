import numpy as np
import cv2

class Gabor:

        def __init__(self, img):
            tup = cv2.findContours(cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.__gblurimg = tup[0]
            print(self.__gblurimg)
            self.__contours = tup[1]
            self.__hierarchy = tup[2]
            self.__momLstForConts = self.__getMoments()
            self.__centroidLstForConts = self.__getCentroidOfCnts()
            (self.__arLstForConts, self.__periLstForConts) = self.__getAreaNPeriOfCnts()
            self.__selecCntImg = self.__getContourImg(imtype='color')
            (self.__imgcovrect, self.__minEdge) = self.__getBoundingRectRotated(imtype='color')
            (self.__imgcovcirc, self.__rad) = self.__getMinEncCirc(imtype='color')
            cv2.namedWindow('1', cv2.WINDOW_NORMAL)
            cv2.imshow('1', self.__gblurimg)
            cv2.waitKey(0)
            cv2.namedWindow('2', cv2.WINDOW_NORMAL)
            cv2.imshow('2', self.__selecCntImg)
            cv2.waitKey(0)
            cv2.namedWindow('3', cv2.WINDOW_NORMAL)
            cv2.imshow('3', self.__imgcovrect)
            cv2.waitKey(0)
            cv2.namedWindow('4', cv2.WINDOW_NORMAL)
            cv2.imshow('4', self.__imgcovcirc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        def __getMoments(self):
            moments = []
            for c in self.__contours:
                moments.append(cv2.moments(c))
            print(moments)
            return moments

        def __getCentroidOfCnts(self):
            centroids = []
            for mlc in self.__momLstForConts:
                coorX = int(mlc['m10'] / mlc['m00'])
                coorY = int(mlc['m01'] / mlc['m00'])
                centroids.append((coorX, coorY))
            print(centroids)
            return centroids

        def __getAreaNPeriOfCnts(self):
            areas = []
            peri = []
            for c in self.__contours:
                areas.append(cv2.contourArea(c))
                peri.append(cv2.arcLength(c, True))
            print(areas, peri)
            return (areas, peri)

        def __getContourImg(self, imtype='gray', cnt=0):
            if (imtype == 'gray'):
                return cv2.drawContours((self.__gblurimg).copy(), [self.__contours[cnt]], 0, 255, 2, cv2.LINE_AA)
            else:
                tmp = cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR)
                return cv2.drawContours(tmp, [self.__contours[cnt]], 0, (0,255,0), 2, cv2.LINE_AA)

        def __getBoundingRectRotated(self, imtype='gray', cnt=0):
            rect = cv2.minAreaRect(self.__contours[cnt])
            mean_edge = ((rect[1])[0] + ((rect[1])[0])) / 2
            if (imtype == 'gray'):
                return (cv2.drawContours((self.__gblurimg).copy(), [np.int0(cv2.boxPoints(rect))], 0, 255, 2, cv2.LINE_AA), mean_edge)
            else:
                return (cv2.drawContours(cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR), [np.int0(cv2.boxPoints(rect))], 0, (0,255,0), 2, cv2.LINE_AA), mean_edge)

        def __getMinEncCirc(self, imtype='gray', cnt=0):
            (x, y), radius = cv2.minEnclosingCircle(self.__contours[cnt])
            center = (int(x), int(y))
            radius = int(radius)
            if (imtype == 'gray'):
                return (cv2.circle((self.__gblurimg).copy(), center, radius, 255, 2, cv2.LINE_AA), radius)
            else:
                return (cv2.circle(cv2.cvtColor(self.__gblurimg, cv2.COLOR_GRAY2BGR), center, radius, (0,255,0), 2, cv2.LINE_AA), radius)

        def __generateAsymmetryIndex(self, totar, idx=0):
            return ((self.__arLstForConts[idx] / totar) * 100)

        def __generateCompactIndex(self, idx=0):
            return (np.power(self.__periLstForConts[0], 2) / (4 * np.pi * self.__arLstForConts[0]))

        def __generateFractalDimension(self):
            return (np.log(self.__minEdge) / np.log(1 / self.__minEdge))

        def __calculateDiameter(self):
            return ()



