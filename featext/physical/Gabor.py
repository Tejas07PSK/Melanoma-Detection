import numpy as np
import cv2

class Gabor:

        def __init__(self, img):
            tup = cv2.findContours(cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.__gblurimg = tup[0]
            self.__contours = tup[1]
            self.__hierarchy = tup[2]
            self.__momLstForConts = self.__getMoments(self.__contours)
            tmp = cv2.drawContours(tmp, tup[1], -1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.moments((tup[1])[0])

        def __getMoments(self, contours):
            moments = []
            for m in contours:
                moments.append(cv2.moments(m))
            print(moments)
            return moments

