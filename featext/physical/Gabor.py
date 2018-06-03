import numpy as np
import cv2

class Gabor:

        def __init__(self, img):
            tup = cv2.findContours(cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.__gblurimg = tup[0]
            self.__contours = tup[1]
            self.__hierarchy = tup[2]
            self.__momLstForConts = self.__getMoments()
            self.__centroidLstForConts = self.__getCentroidOfCnts()
            (self.__arLstForConts, self.__periLstForConts) = self.__getAreaNPeriOfCnts()
            self.__selecCntImg = self.__getContourImg()
            self.__imgcovrect = self.__getBoundingRectRotated()


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
                return cv2.drawContours(self.__gblurimg, [self.__contours[cnt]], 0, 255, 2, cv2.LINE_AA)
            else:
                return cv2.drawContours(self.__gblurimg, [self.__contours[cnt]], 0, (0,255,0), 2, cv2.LINE_AA)

        def __getBoundingRectRotated(self, imtype='gray', cnt=0):
            rect = cv2.minAreaRect(self.__contours[cnt])
            print(rect)
            box = cv2.boxPoints(rect)
            print(box)
            box = np.int0(box)
            print(box)
            if (imtype == 'gray'):
                return cv2.drawContours(self.__gblurimg, [box], 0, 255, 2, cv2.LINE_AA)
            else:
                return cv2.drawContours(self.__gblurimg, [box], 0, (0,255,0), 2, cv2.LINE_AA)

