import cv2
from preprocessing import Prep as p

obj=p.Prep('melanoma.jpg')


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


showColImg()
showBinImg()
showGrayImg()
showInvertedGrayImg()
showSegmentedColorImg()
showSegmentedGrayImg()
