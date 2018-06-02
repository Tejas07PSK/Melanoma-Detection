import numpy as np
import cv2

class Gabor:

        def __init__(self, img):
            kernel = cv2.getGaussianKernel(ksize=5, sigma=0, ktype=float)
            print(kernel)
            tup = cv2.findContours(cv2.GaussianBlur(img, kernel, sigmaX=0, sigmaY=0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(tup[0])
            print(tup[1])
            print(tup[3])

