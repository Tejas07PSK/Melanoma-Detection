import numpy as np
import cv2

class Gabor:

        def __init__(self, img):
            tmp = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
            tup = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.namedWindow('gb', cv2.WINDOW_NORMAL)
            cv2.imshow('gb', tmp)
            cv2.waitKey(0)
            print(tup[0])
            print((tup[1])[0])
            print((tup[1])[1])
            print((tup[1])[2])
            print(tup[2])

