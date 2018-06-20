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
from mlmodels import DecisionSurfacePlotter as DSP

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
         os.makedirs('results/dataset/' + restype + '/' + str(i))
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

def __createAndTrainMlModels():
    dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    CLF.Classifiers(featureset=dset['featureset'], target=__convertTargetTypeToInt(dset['result']), mode='train', path='mlmodels/')
    print("Training successfully completed!!! \n")

def getTestImages():
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
    np.savez('testcase', dset=dset, featnames=featnames)

def __convertTargetTypeToInt(arr):
    cvt_arr = np.zeros((arr.size,), int, 'C')
    for i in range(0, arr.size, 1):
        if (arr[i] == 'malignant'):
            cvt_arr[i] = 1
        elif (arr[i] == 'negative'):
            cvt_arr[i] = -1
        else:
            continue
    return cvt_arr

def __convertTargetTypeToStr(arr):
    cvt_arr = np.empty((arr.size,), object, 'C')
    for i in range(0, arr.size, 1):
        if (int(np.round(arr[i])) >= 1):
            cvt_arr[i] = 'malignant'
        elif (int(np.round(arr[i])) <= -1):
            cvt_arr[i] = 'negative'
        elif (int(np.round(arr[i])) == 0):
            cvt_arr[i] = 'benign'
        else:
            pass
    return cvt_arr

def predictFromSavedTestCase():
    clasfobj = CLF.Classifiers(path='mlmodels/')
    dset, featnames = (np.load('testcase.npz'))['dset'], (np.load('testcase.npz'))['featnames']
    print(featnames)
    print(dset)
    print(dset['featureset'])
    print(dset['result'])
    print("\n")
    print("Now predicting results : \n \n")
    pred_res = clasfobj.predicto(dset['featureset'], __convertTargetTypeToInt(dset['result']))
    return pred_res

def __printPredResWithProperFormatting(predres, type='RFC'):
    if (type == 'SVM'):
        print(" FOR SVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['SVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['SVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'SVR'):
        print(" FOR SVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['SVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['SVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'NuSVM'):
        print(" FOR NuSVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['NuSVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['NuSVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['NuSVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'NuSVR'):
        print(" FOR NuSVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['NuSVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['NuSVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['NuSVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'LinSVM'):
        print(" FOR LinSVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['LinSVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['LinSVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['LinSVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'LinSVR'):
        print(" FOR LinSVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['LinSVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['LinSVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['LinSVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'MLPC'):
        print(" FOR MLPC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['MLPC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['MLPC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['MLPC'])['Accuracy'] * 100) + "\n")
    elif (type == 'MLPR'):
        print(" FOR MLPR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['MLPR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['MLPR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['MLPR'])['Accuracy'] * 100) + "\n")
    elif (type == 'DTC'):
        print(" FOR DTC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['DTC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['DTC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['DTC'])['Accuracy'] * 100) + "\n")
    elif (type == 'DTR'):
        print(" FOR DTR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['DTR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['DTR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['DTR'])['Accuracy'] * 100) + "\n")
    elif (type == 'RFC'):
        print(" FOR RFC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['RFC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['RFC'])['Accuracy'] * 100) + "\n")
    else:
        print(" FOR RFR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['RFR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['RFR'])['Accuracy'] * 100) + "\n")

def __printfeatsfromfile(fl='testcase.npz'):
    dset, featnames = (np.load(fl))['dset'], (np.load(fl))['featnames']
    for i in range(0, ((dset['featureset']).shape)[0], 1):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
        print("Printing features for stored image - %d \n" % i)
        for j in range(0, ((dset['featureset']).shape)[1], 1):
            print(" %s \t -:- \t %f \n" % (str(featnames[j]), (dset['featureset'])[i,j]))
        print("Image is of type --- %s \n" % (str((dset['result'])[i])).upper())
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

def __listFilesInDir(loc):
     return (flnm for flnm in os.listdir(loc) if os.path.isfile(os.path.join(loc, flnm)))

def main_menu():
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^_______WELCOME TO THE MELANOMA-PREDICTION PROGRAM_______^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n")
    print("\t This is a nascent approch towards detecting Melanoma-Skin-Lesion, using OpenCV, NumPY and SciKit in Python Programming Language. \n")
    print("\t This project utilizes some of the core concepts of \'DIGITAL IMAGE PROCESSING\' & \'MACHINE LEARNING\'. \n")
    print("\t This program can either categorize the cancerous-lesion as Malignant, Benign or Negative. \n")
    print("\t 1.Create training-dataset from the images of known ->MELANOMA<- types!! \n")
    print("\t 2.Train classifiers and regressors on created training-dataset!! \n")
    print("\t 3.Create testing-dataset from supervised images in temp folder!! \n")
    print("\t 4.Predict results from \'testcase.npz\'!! \n")
    print("\t 5.Print feature-descriptors of images strored in numpy files, training or testcase!! \n")
    print("\t 6.Plot Classifier graphs!! \n")
    print("\t 7.Add the featuresets in \'testcase.npz\' to \'dataset.npz\' to make mlmodels more accurate!! \n")
    print("\t Enter \'e\' to exit!! \n")
    while (True):
       c = str(input("Enter your choice - \n"))
       if (c == '1'):
           print("If you see a results folder in the root directory of the project, delete the \'dataset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your corresponding images in the \'images\' folder under the malignanat, benign or negative directories. \n")
           print("If you haven't already made the directories, please make them and place the corrseponding images. \n")
           print("The image filenames names must be numeric starting from 0 in sequence under each category folder. \n")
           print("Eg. - 0.jpg, 1.jpg, 2.jpg, ..... etc \n")
           print("You must provide images under each category!!! \n")
           input("Just press any key when your are ready : \n")
           createDataSet("malignant", int(input("Enter the number of images you placed under the \'images/malignant\' directory - \n")))
           createDataSet("benign", int(input("Enter the number of images you placed under the \'images/benign\' directory - \n")))
           createDataSet("negative", int(input("Enter the number of images you placed under the \'images/negative\' directory - \n")))
           print("Training-dataset successfully generated!! \n")
           print("This dataset consists of the features-array of the corresponding images and their classified types. \n")
           print("All results are stored in the file \'dataset.npz\' \n")
           print("Total training-image count : %d \n" % imgcount)
       elif (c == '2'):
           print("Now we'll train our various classifiers and regressors on the training data stored in the \'dataset.npz\' numpy file. \n")
           print("All machine-learning models will be saved in individual .pkl files under the \'mlmodels\' python-package. \n")
           __createAndTrainMlModels()
           print("Training is now complete!! \n")
       elif (c == '3'):
           print("If you see a results folder in the root directory of the project, delete the \'testset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your test-images in the \'temp\' folder. \n")
           print("If you haven't already made the directories, please make them and place the test-images. \n")
           input("Just press any key when your are ready : \n")
           getTestImages()
           print("Testing-dataset successfully generated!! \n")
           print("This dataset consists of the features-array of the test images and their supervised-classified types. \n")
           print("All results are stored in the file \'testset.npz\' \n")
       elif (c == '4'):
           print("This will predict results from \'testcase.npz\' and also calculate the prediction accuracy of the individual models. \n")
           pred_res = predictFromSavedTestCase()
           while (True):
              type = str(input('Select Classifier/Regressor acronym : \n'))
              if (type in pred_res):
                    __printPredResWithProperFormatting(pred_res, type)
              else:
                  __printPredResWithProperFormatting(pred_res)
                  break
           print("\n \n")
       elif (c == '5'):
           print("This option prints the features of the images stored in the testcase.npz file, by default. \n")
           print("You can also print the values stored in the dataset.npz file, just pass this file-name as an argument in the function below. \n")
           print("Before you print the feature-contents, make sure that you have previously generated the dataset.npz and testcase.npz files. \n")
           __printfeatsfromfile(str(input('Enter the filename : \n')))
           print("PRINTING COMPLETE!!! \n")
       elif (c == '6'):
           print("\t Before you proceed, make sure that you have previously generated the \'dataset.npz\' and is existing in the root directory of the project!!! \n")
           dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
           print("\t Given below are the set of features, along with their corresponding indexes. \n")
           for count in range(0, featnames.size, 1):
               print(str(count)+". "+str(featnames[count])+" \n")
           print("You have to select a combination of any two features!! \n")
           print("You can also enter multiple combinations, in such case they will be appended to a list!! \n")
           flist = []
           fnlist = []
           while(True):
               feat_corrs = [int(input('Enter index of first feature ... \n')), int(input('Enter index of second feature ... \n'))]
               flist.append(feat_corrs)
               fnlist.append([featnames[feat_corrs[0]], featnames[feat_corrs[1]]])
               if (str(input('Do you want to add more feature combinations?? - (y/n) \n')) == 'y'):
                   continue
               else:
                   break
           DSP.plotForAll(dset['featureset'], __convertTargetTypeToInt(dset['result']), flist, fnlist)
           print("DONE!!! \n")
       elif (c == '7'):
           nfls = [__listFilesInDir("images/" + str(cls)) for cls in ('benign', 'malignant', 'negative')]
           print(nfls)
           """trainset, testset = (np.load('dataset.npz'))['dset'], (np.load('testcase.npz'))['dset']
           for feat, index in zip(testset, range(0, testset.size, 1)):
               if (feat[1] == 'benign'):
                   __listFilesInDir("images/" + feat[1])"""
       else:
           print("Thanks For Using This Program!!!")
           print("Now Exiting.")
           break

main_menu()



