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
            tmp = cv2.drawContours(tmp, tup[1], -1, (0, 255, 0), 3, cv2.LINE_AA)

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

        def __

