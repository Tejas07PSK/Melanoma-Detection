import cv2
import numpy as np
import pathlib
import os
from preprocessing import Prep as p
from featext.texture import Haralick as har
from featext.texture import Tamura as tam
from featext.texture import King as k
from featext.physical import Gabor as g
from mlmodels import Classifiers as CLF

imgcount = 0

def showColImg(obj, index, loc):
    #cv2.namedWindow('imgcol' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('imgcol' + index, obj.getActImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc+'imgcol'+index+'.jpg'), obj.getActImg())

def showGrayImg(obj, index, loc):
    #cv2.namedWindow('imggray' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('imggray' + index, obj.getGrayImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'imggray' + index + '.jpg'), obj.getGrayImg())

def showInvertedGrayImg(obj, index, loc):
    #cv2.namedWindow('imggrayinvrt' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('imggrayinvrt' + index, obj.getInvrtGrayImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'imggrayinvrt' + index + '.jpg'), obj.getInvrtGrayImg())

def showBinImg(obj, index, loc):
    #cv2.namedWindow('imgbin' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('imgbin' + index, obj.getBinaryImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'imgbin' + index + '.jpg'), obj.getBinaryImg())

def showSegmentedColorImg(obj, index, loc):
    #cv2.namedWindow('segimgcol' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('segimgcol' + index, obj.getSegColImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'segimgcol' + index + '.jpg'), obj.getSegColImg())

def showSegmentedGrayImg(obj, index, loc):
    #cv2.namedWindow('segimggray' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('segimggray' + index, obj.getSegGrayImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'segimggray' + index + '.jpg'), obj.getSegGrayImg())

def showPrewittHorizontalImg(feobj2, index, loc):
    #cv2.namedWindow('PrewittX' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('PrewittX' + index, feobj2.getPrewittHorizontalEdgeImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'PrewittX' + index + '.jpg'), feobj2.getPrewittHorizontalEdgeImg())

def showPrewittVerticalImg(feobj2, index, loc):
    #cv2.namedWindow('PrewittY' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('PrewittY' + index, feobj2.getPrewittVerticalEdgeImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'PrewittY' + index + '.jpg'), feobj2.getPrewittVerticalEdgeImg())

def showPrewittCOmbinedImg(feobj2, index, loc):
    #cv2.namedWindow('PrewittIMG' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('PrewittIMG' + index, feobj2.getCombinedPrewittImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'PrewittIMG' + index + '.jpg'), feobj2.getCombinedPrewittImg())

def showGaussBlurredSegImg(feobj3, index, loc):
    #cv2.namedWindow('gblurimg' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('gblurimg' + index, feobj3.getGaussianBlurredImage())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'gblurimg' + index + '.jpg'), feobj3.getGaussianBlurredImage())

def showSelectedContourImg(feobj3, index, loc):
    #cv2.namedWindow('slccntimg' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('slccntimg' + index, feobj3.getSelectedContourImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'slccntimg' + index + '.jpg'), feobj3.getSelectedContourImg())

def showBoundingRectImg(feobj3, index, loc):
    #cv2.namedWindow('bndrectimg' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('bndrectimg' + index, feobj3.getBoundingRectImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'bndrectimg' + index + '.jpg'), feobj3.getBoundingRectImg())

def showBoundingCircImg(feobj3, index, loc):
    #cv2.namedWindow('bndcircimg' + index, cv2.WINDOW_NORMAL)
    #cv2.imshow('bndcircimg' + index, feobj3.getBoundedCircImg())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite((loc + 'bndcircimg' + index + '.jpg'), feobj3.getBoundedCircImg())

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

def showKingsFeatures(feobj4):
    print("->.->.->.->.->.->.->.->.->.KING'S TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    print("\n")
    print(feobj4.getNGTDM())
    print("\n")
    print("King's-Coarseness of seg gray img %f \n" % feobj4.getKingsCoarseness())
    print("King's-Contrast of seg gray img %f \n" % feobj4.getKingsContrast())
    print("King's-Busyness of seg gray img %f \n" % feobj4.getKingsBusyness())
    print("King's-Complexity of seg gray img %f \n" % feobj4.getKingsComplexity())
    print("King's-Strength of seg gray img %f \n" % feobj4.getKingsStrength())

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
    print("------------------+++++++++++++============FOR %s SET==============++++++++++++++---------------------- \n" % restype.upper())
    if (((pathlib.Path('dataset.npz')).exists() == True) & ((pathlib.Path('dataset.npz')).is_file() == True)):
        dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    else:
        dset = np.empty(0, dtype=np.dtype([('featureset', float, (34,)), ('result', object)]), order='C')
        featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
    for i in range(0, img_num, 1):
         global imgcount
         print("\t _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ \t \n")
         print("Iterating for image - %d \n" % i)
         print("\t _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ \t \n")
         obj = p.Prep('images/' + restype + '/' + str(i) + '.jpg')
         feobj = har.HarFeat(obj.getSegGrayImg())
         feobj2 = tam.TamFeat(obj.getSegGrayImg())
         feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
         feobj4 = k.KingFeat(obj.getSegGrayImg())
         showColImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showGrayImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showInvertedGrayImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showBinImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showSegmentedColorImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showSegmentedGrayImg(obj, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showPrewittHorizontalImg(feobj2, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showPrewittVerticalImg(feobj2, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showPrewittCOmbinedImg(feobj2, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showGaussBlurredSegImg(feobj3, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showSelectedContourImg(feobj3, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showBoundingRectImg(feobj3, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showBoundingCircImg(feobj3, str(imgcount), 'results/dataset/' + restype + '/' + str(i) + '/')
         showHaralickFeatures(feobj)
         showTamuraFeatures(feobj2)
         showKingsFeatures(feobj4)
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
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsCoarseness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsBusyness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsComplexity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsStrength(), 0)
         dset = np.insert(dset, dset.size, (featarr, restype), 0)
         imgcount = imgcount + 1
    print(featnames)
    print(dset)
    print(dset['featureset'])
    print(dset['result'])
    print("\n")
    np.savez('dataset', dset=dset, featnames=featnames)
    __createAndTrainMlModels()

def __createAndTrainMlModels():
    dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    CLF.Classifiers(featureset=dset['featureset'], target=dset['result'], mode='train', path='mlmodels/')
    print("Training successfully completed!!! \n")

"""createDataSet("malignant", 8)
createDataSet("benign", 8)
createDataSet("negative", 8)
print(imgcount)"""

def getTestImages():
    clasfobj = CLF.Classifiers(path='mlmodels/')
    count = 0
    dset = np.empty(0, dtype=np.dtype([('featureset', float, (34,)), ('result', object)]), order='C')
    featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
    while(True):
        os.makedirs('results/testset/' + str(count))
        print("\t _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ \t \n")
        print("Iterating for image - %d \n" % count)
        print("\t _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ \t \n")
        imgnm = str(input('Enter image name : \n'))
        obj = p.Prep('temp/' + imgnm)
        feobj = har.HarFeat(obj.getSegGrayImg())
        feobj2 = tam.TamFeat(obj.getSegGrayImg())
        feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
        feobj4 = k.KingFeat(obj.getSegGrayImg())
        showColImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showGrayImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showInvertedGrayImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showBinImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showSegmentedColorImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showSegmentedGrayImg(obj, str(count), 'results/testset/' + str(count) + '/')
        showPrewittHorizontalImg(feobj2, str(count), 'results/testset/' + str(count) + '/')
        showPrewittVerticalImg(feobj2, str(count), 'results/testset/' + str(count) + '/')
        showPrewittCOmbinedImg(feobj2, str(count), 'results/testset/' + str(count) + '/')
        showGaussBlurredSegImg(feobj3, str(count), 'results/testset/' + str(count) + '/')
        showSelectedContourImg(feobj3, str(count), 'results/testset/' + str(count) + '/')
        showBoundingRectImg(feobj3, str(count), 'results/testset/' + str(count) + '/')
        showBoundingCircImg(feobj3, str(count), 'results/testset/' + str(count) + '/')
        showHaralickFeatures(feobj)
        showTamuraFeatures(feobj2)
        showKingsFeatures(feobj4)
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
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsCoarseness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsBusyness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsComplexity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsStrength(), 0)
        dset = np.insert(dset, dset.size, (featarr, str(input('Enter your result : \n'))), 0)
        count = count + 1
        if(str(input('Do you want to enter more images?? \n')) == 'Y'):
            continue
        else:
            break
    print(featnames)
    print(dset)
    print(dset['featureset'])
    print(dset['result'])
    print("\n")
    if (str(input('Do you want to save this testcase for later predictions?? \n')) == 'Y'):
        np.savez('testcase', dset=dset, featnames=featnames)
    print("Now predicting results : \n \n")
    pred_res = clasfobj.predicto(dset['featureset'], dset['result'])
    __printPredResWithProperFormatting(pred_res)
    print('\n \n')

def predictFromSavedTestCase():
    clasfobj = CLF.Classifiers(path='mlmodels/')
    dset, featnames = (np.load('testcase.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    print(featnames)
    print(dset)
    print(dset['featureset'])
    print(dset['result'])
    print("\n")
    print("Now predicting results : \n \n")
    pred_res = clasfobj.predicto(dset['featureset'], dset['result'])
    __printPredResWithProperFormatting(pred_res)
    print("\n \n")

def __printPredResWithProperFormatting(predres):
    print("Prediction Results SVM - " + str((predres['SVM'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['SVM'])['Accuracy'] * 100) + "\n")
    print("Prediction Results NuSVM - " + str((predres['NuSVM'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['NuSVM'])['Accuracy'] * 100) + "\n")
    print("Prediction Results LinSVM - " + str((predres['LinSVM'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['LinSVM'])['Accuracy'] * 100) + "\n")
    print("Prediction Results MLPC - " + str((predres['MLPC'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['MLPC'])['Accuracy'] * 100) + "\n")
    print("Prediction Results DTC - " + str((predres['DTC'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['DTC'])['Accuracy'] * 100) + "\n")
    print("Prediction Results SVM - " + str((predres['RFC'])['Prediction Results']) + "and Accuracy Prediction - " + str((predres['RFC'])['Accuracy'] * 100) + "\n")

getTestImages()


