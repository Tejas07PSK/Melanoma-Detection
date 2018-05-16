import cv2
import numpy as np
from preprocessing import Prep as p
from featext import FeatExt as fe

obj = p.Prep('melanoma.jpg')
arr1 = obj.getArrayOfGrayLevelsWithFreq(obj.getSegGrayImg())
arr2 = obj.getArrayOfGrayLevelsWithFreq(obj.getBinaryImg())
arr3 = obj.getArrayOfGrayLevelsWithFreq(obj.getGrayImg())
arr4 = obj.getArrayOfGrayLevelsWithFreq(obj.getInvrtGrayImg())
feobj = fe.FeatExt(obj.getSegGrayImg(), arr1)

def showColImg():
    cv2.namedWindow('imgcol', cv2.WINDOW_NORMAL)
    cv2.imshow('imgcol', obj.getActImg())
    cv2.waitKey(0)


def showGrayImg():
    cv2.namedWindow('imggray', cv2.WINDOW_NORMAL)
    cv2.imshow('imggray', obj.getGrayImg())
    cv2.waitKey(0)


def showInvertedGrayImg():
    cv2.namedWindow('imggrayinvrt', cv2.WINDOW_NORMAL)
    cv2.imshow('imggrayinvrt', obj.getInvrtGrayImg())
    cv2.waitKey(0)


def showBinImg():
    cv2.namedWindow('imgbin', cv2.WINDOW_NORMAL)
    cv2.imshow('imgbin', obj.getBinaryImg())
    cv2.waitKey(0)


def showSegmentedColorImg():
    cv2.namedWindow('segimgcol', cv2.WINDOW_NORMAL)
    cv2.imshow('segimgcol', obj.getSegColImg())
    cv2.waitKey(0)


def showSegmentedGrayImg():
    cv2.namedWindow('segimggray', cv2.WINDOW_NORMAL)
    cv2.imshow('segimggray', obj.getSegGrayImg())
    cv2.waitKey(0)

def showGLCM():
    np.savetxt('glcm.txt', feobj.getGLCM(), '%u', encoding="UTF-8")
    np.savetxt('glvl1.txt', arr1, '(%u, %u)', encoding="UTF-8")
    np.savetxt('glvl2.txt', arr2, '(%u, %u)', encoding="UTF-8")
    np.savetxt('glvl3.txt', arr3, '(%u, %u)', encoding="UTF-8")
    np.savetxt('glvl4.txt', arr4, '(%u, %u)', encoding="UTF-8")
    print(feobj.getGLCM())

def noofpixles(arr):
    sum = np.uint32(0)
    for x in arr:
        sum = sum + np.uint32((x[1])[0])
    return sum

showGLCM()
showColImg()
showGrayImg()
showInvertedGrayImg()
showBinImg()
showSegmentedColorImg()
showSegmentedGrayImg()

print("No of pixels in seg gray img %u \n" % noofpixles(arr1))
print("No of pixels in bin img %u \n" % noofpixles(arr2))
print("No of pixels in gray img %u \n" % noofpixles(arr3))
print("No of pixels in inv gray img %u \n" % noofpixles(arr4))

