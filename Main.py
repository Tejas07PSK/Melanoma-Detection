import cv2
import numpy as np
from preprocessing import Prep as p
from featext.texture import Haralick as har
from featext.texture import Tamura as tam
from featext.physical import Gabor as g

def showColImg(obj, index):
    cv2.namedWindow('imgcol' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('imgcol' + index, obj.getActImg())
    #cv2.waitKey(0)

def showGrayImg(obj, index):
    cv2.namedWindow('imggray' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('imggray' + index, obj.getGrayImg())
    #cv2.waitKey(0)

def showInvertedGrayImg(obj, index):
    cv2.namedWindow('imggrayinvrt' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('imggrayinvrt' + index, obj.getInvrtGrayImg())
    #cv2.waitKey(0)

def showBinImg(obj, index):
    cv2.namedWindow('imgbin' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('imgbin' + index, obj.getBinaryImg())
    #cv2.waitKey(0)

def showSegmentedColorImg(obj, index):
    cv2.namedWindow('segimgcol' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('segimgcol' + index, obj.getSegColImg())
    #cv2.waitKey(0)

def showSegmentedGrayImg(obj, index):
    cv2.namedWindow('segimggray' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('segimggray' + index, obj.getSegGrayImg())
    #cv2.waitKey(0)

def showPrewittHorizontalImg(feobj2, index):
    cv2.namedWindow('PrewittX' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('PrewittX' + index, feobj2.getPrewittHorizontalEdgeImg())
    #cv2.waitKey(0)

def showPrewittVerticalImg(feobj2, index):
    cv2.namedWindow('PrewittY' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('PrewittY' + index, feobj2.getPrewittVerticalEdgeImg())
    #cv2.waitKey(0)

def showPrewittCOmbinedImg(feobj2, index):
    cv2.namedWindow('PrewittIMG' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('PrewittIMG' + index, feobj2.getCombinedPrewittImg())
    #cv2.waitKey(0)

def showGaussBlurredSegImg(feobj3, index):
    cv2.namedWindow('gblurimg' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('gblurimg' + index, feobj3.getGaussianBlurredImage())
    #cv2.waitKey(0)

def showSelectedContourImg(feobj3, index):
    cv2.namedWindow('slccntimg' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('slccntimg' + index, feobj3.getSelectedContourImg())
    #cv2.waitKey(0)

def showBoundingRectImg(feobj3, index):
    cv2.namedWindow('bndrectimg' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('bndrectimg' + index, feobj3.getBoundingRectImg())
    #cv2.waitKey(0)

def showBoundingCircImg(feobj3, index):
    cv2.namedWindow('bndcircimg' + index, cv2.WINDOW_NORMAL)
    cv2.imshow('bndcircimg' + index, feobj3.getBoundedCircImg())
    #cv2.waitKey(0)

def showGLCM(feobj):
    print(feobj.getGLCM())

def showHaralickFeatures(feobj):
    print("->.->.->.->.->.->.->.->.->.HARALICK TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    showGLCM(feobj)
    print("\n")
    print("Angular Second Moment-ASM of seg gray img %f \n" % feobj.getAngularSecondMomentASM())
    print("Energy of seg gray img %f \n" % feobj.getEnergy())
    print("Entropy of seg gray img %f \n" % feobj.getEntropy())
    print("Contrast of seg gray img %f \n" % feobj.getContrast())
    print("Homogeneity of seg gray img %f \n" % feobj.getHomogeneity())
    print("Directional-Moment of seg gray img %f \n" % feobj.getDm())
    print("Correlation of seg gray img %f \n" % feobj.getCorrelation())
    print("Haralick-Correlation of seg gray img %f \n" % feobj.getHarCorrelation())
    print("Cluster-Shade of seg gray img %f \n" % feobj.getClusterShade())
    print("Cluster-Prominence of seg gray img %f \n" % feobj.getClusterProminence())
    print("Moment1 of seg gray img %f \n" % feobj.getMoment1())
    print("Moment2 of seg gray img %f \n" % feobj.getMoment2())
    print("Moment3 of seg gray img %f \n" % feobj.getMoment3())
    print("Moment4 of seg gray img %f \n" % feobj.getMoment4())
    print("Differential-ASM of seg gray img %f \n" % feobj.getDasm())
    print("Differential-Mean of seg gray img %f \n" % feobj.getDmean())
    print("Differential-Entropy of seg gray img %f \n" % feobj.getDentropy())

def showTamuraFeatures(feobj2):
    print("->.->.->.->.->.->.->.->.->.TAMURA TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    print("Tamura-Coarseness of seg gray img %f \n" % feobj2.getCoarseness())
    print("Tamura-Contrast of seg gray img %f \n" % feobj2.getContrast())
    print("Tamura-Kurtosis of seg gray img %f \n" % feobj2.getKurtosis())
    print("Tamura-LineLikeness of seg gray img %f \n" % feobj2.getLineLikeness())
    print("Tamura-Directionality of seg gray img %f \n" % feobj2.getDirectionality())
    print("Tamura-Regularity of seg gray img %f \n" % feobj2.getRegularity())
    print("Tamura-Roughness of seg gray img %f \n" % feobj2.getRoughness())

def showGaborPhysicalFeatures(feobj3):
    print("->.->.->.->.->.->.->.->.->.GABOR PHYSICAL FEATURES OF LESION.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    print("List of Contour-Points ::: \n")
    print(feobj3.getListOfContourPoints())
    print("\n")
    print("Hierarchy of extracted contours ::: \n")
    print(feobj3.getHierarchyOfContours())
    print("\n")
    print("List of moments for corresponding-contours ::: \n")
    print(feobj3.getListOfMomentsForCorrespondingContours())
    print("\n")
    print("List of centroids for corresponding-contours ::: \n")
    print(feobj3.getListOfCentroidsForCorrespondingContours())
    print("\n")
    print("List of areas for corresponding-contours ::: \n")
    print(feobj3.getListOfAreasForCorrespondingContours())
    print("\n")
    print("List of perimeters for corresponding-contours ::: \n")
    print(feobj3.getListOfPerimetersForCorrespondingContours())
    print("\n")
    print("Mean_Edge of covering rectangle for lesion img %f \n" % feobj3.getMeanEdgeOfCoveringRect())
    print("Bounded_Circle radius %f \n" % feobj3.getBoundedCircRadius())
    print("Asymmetry-Index of lesion %f \n" % feobj3.getAsymmetryIndex())
    print("Compact-Index of lesion %f \n" % feobj3.getCompactIndex())
    print("Fractal-Dimension of lesion %f \n" % feobj3.getFractalDimension())
    print("Diameter of lesion %f \n" % feobj3.getDiameter())
    print("Color-Variance of lesion %f \n" % feobj3.getColorVariance())


def createDataSet(restype, img_num):
    dset = np.empty(0, dtype=np.dtype([('featureset', float, (29,)), ('result', str)]), order='C')
    for i in range(0, img_num, 1):
         print("Iterating for image - %d \n", i)
         index = str(i)
         obj = p.Prep('images/' + restype + '/' + index + '.jpg')
         feobj = har.HarFeat(obj.getSegGrayImg())
         feobj2 = tam.TamFeat(obj.getSegGrayImg())
         feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
         showColImg(obj, index)
         showGrayImg(obj, index)
         showInvertedGrayImg(obj, index)
         showBinImg(obj, index)
         showSegmentedColorImg(obj, index)
         showSegmentedGrayImg(obj, index)
         showPrewittHorizontalImg(feobj2, index)
         showPrewittVerticalImg(feobj2, index)
         showPrewittCOmbinedImg(feobj2, index)
         showGaussBlurredSegImg(feobj3, index)
         showSelectedContourImg(feobj3, index)
         showBoundingRectImg(feobj3, index)
         showBoundingCircImg(feobj3, index)
         showHaralickFeatures(feobj)
         showTamuraFeatures(feobj2)
         showGaborPhysicalFeatures(feobj3)
         featarr = np.empty(0, dtype=float, order='C')
         featarr = np.insert(featarr, featarr.size, feobj.getAngularSecondMomentASM(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getEnergy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getEntropy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getHomogeneity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDm(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getCorrelation(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getHarCorrelation(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getClusterShade(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getClusterProminence(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment1(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment2(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment3(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment4(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDasm(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDmean(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDentropy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getCoarseness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getKurtosis(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getLineLikeness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getDirectionality(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getRegularity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getRoughness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getAsymmetryIndex(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getCompactIndex(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getFractalDimension(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getDiameter(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getColorVariance(), 0)
         dset = np.insert(dset, dset.size, (featarr, restype), 0)
    print(dset)
    print(dset['featureset'])
    print(dset['result'])
    np.save('dataset', dset, allow_pickle=True, fix_imports=True)

createDataSet('malignant', 2)
cv2.waitKey(0)
cv2.destroyAllWindows()