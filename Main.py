import cv2
import numpy as np
import pathlib
import os
from shutil import copyfile
from preprocessing import Prep as p
from featext.texture import Haralick as har
from featext.texture import Tamura as tam
from featext.texture import King as k
from featext.physical import Gabor as g
from mlmodels import Classifiers as CLF
from mlmodels import DecisionSurfacePlotter as DSP

imgcount = 0

def __showImages(lstofimgs):
    for tpls in lstofimgs:
        cv2.namedWindow(tpls[1], cv2.WINDOW_NORMAL)
        cv2.imshow(tpls[1], tpls[0])
        if (tpls[2] != None):
            cv2.imwrite(tpls[2], tpls[0])
        else:
            continue
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __showGLCM(feobj):
    print(feobj.getGLCM())

def __showHaralickFeatures(feobj):
    print("->.->.->.->.->.->.->.->.->.HARALICK TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    __showGLCM(feobj)
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

def __showTamuraFeatures(feobj2):
    print("->.->.->.->.->.->.->.->.->.TAMURA TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    print("Tamura-Coarseness of seg gray img %f \n" % feobj2.getCoarseness())
    print("Tamura-Contrast of seg gray img %f \n" % feobj2.getContrast())
    print("Tamura-Kurtosis of seg gray img %f \n" % feobj2.getKurtosis())
    print("Tamura-LineLikeness of seg gray img %f \n" % feobj2.getLineLikeness())
    print("Tamura-Directionality of seg gray img %f \n" % feobj2.getDirectionality())
    print("Tamura-Regularity of seg gray img %f \n" % feobj2.getRegularity())
    print("Tamura-Roughness of seg gray img %f \n" % feobj2.getRoughness())

def __showKingsFeatures(feobj4):
    print("->.->.->.->.->.->.->.->.->.KING'S TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
    print("\n")
    print(feobj4.getNGTDM())
    print("\n")
    print("King's-Coarseness of seg gray img %f \n" % feobj4.getKingsCoarseness())
    print("King's-Contrast of seg gray img %f \n" % feobj4.getKingsContrast())
    print("King's-Busyness of seg gray img %f \n" % feobj4.getKingsBusyness())
    print("King's-Complexity of seg gray img %f \n" % feobj4.getKingsComplexity())
    print("King's-Strength of seg gray img %f \n" % feobj4.getKingsStrength())

def __showGaborPhysicalFeatures(feobj3):
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

def __createDataSet(restype, img_num):
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
         __showImages([(obj.getActImg(), 'imgcol' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'imgcol' + str(imgcount) + '.jpg'),
                       (obj.getGrayImg(), 'imggray' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'imggray' + str(imgcount) + '.jpg'),
                       (obj.getInvrtGrayImg(), 'imggrayinvrt' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'imggrayinvrt' + str(imgcount) + '.jpg'),
                       (obj.getBinaryImg(), 'imgbin' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'imgbin' + str(imgcount) + '.jpg'),
                       (obj.getSegColImg(), 'segimgcol' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'segimgcol' + str(imgcount) + '.jpg'),
                       (obj.getSegGrayImg(), 'segimggray' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'segimggray' + str(imgcount) + '.jpg'),
                       (feobj2.getPrewittHorizontalEdgeImg(), 'PrewittX' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'PrewittX' + str(imgcount) + '.jpg'),
                       (feobj2.getPrewittVerticalEdgeImg(), 'PrewittY' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'PrewittY' + str(imgcount) + '.jpg'),
                       (feobj2.getCombinedPrewittImg(), 'PrewittIMG' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'PrewittIMG' + str(imgcount) + '.jpg'),
                       (feobj3.getGaussianBlurredImage(), 'gblurimg' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'gblurimg' + str(imgcount) + '.jpg'),
                       (feobj3.getSelectedContourImg(), 'slccntimg' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'slccntimg' + str(imgcount) + '.jpg'),
                       (feobj3.getBoundingRectImg(), 'bndrectimg' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'bndrectimg' + str(imgcount) + '.jpg'),
                       (feobj3.getBoundedCircImg(), 'bndcircimg' + str(imgcount), 'results/dataset/' + restype + '/' + str(imgcount) + '/' + 'bndcircimg' + str(imgcount) + '.jpg')])
         __showHaralickFeatures(feobj)
         __showTamuraFeatures(feobj2)
         __showKingsFeatures(feobj4)
         __showGaborPhysicalFeatures(feobj3)
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

def __getTestImages():
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
        __showImages([(obj.getActImg(), 'imgcol' + str(count), 'results/testset/' + str(count) + '/' + 'imgcol' + str(count) + '.jpg'),
                      (obj.getGrayImg(), 'imggray' + str(count), 'results/testset/' + str(count) + '/' + 'imggray' + str(count) + '.jpg'),
                      (obj.getInvrtGrayImg(), 'imggrayinvrt' + str(count), 'results/testset/' + str(count) + '/' + 'imggrayinvrt' + str(count) + '.jpg'),
                      (obj.getBinaryImg(), 'imgbin' + str(count), 'results/testset/' + str(count) + '/' + 'imgbin' + str(count) + '.jpg'),
                      (obj.getSegColImg(), 'segimgcol' + str(count), 'results/testset/' + str(count) + '/' + 'segimgcol' + str(count) + '.jpg'),
                      (obj.getSegGrayImg(), 'segimggray' + str(count), 'results/testset/' + str(count) + '/' + 'segimggray' + str(count) + '.jpg'),
                      (feobj2.getPrewittHorizontalEdgeImg(), 'PrewittX' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittX' + str(count) + '.jpg'),
                      (feobj2.getPrewittVerticalEdgeImg(), 'PrewittY' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittY' + str(count) + '.jpg'),
                      (feobj2.getCombinedPrewittImg(), 'PrewittIMG' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittIMG' + str(count) + '.jpg'),
                      (feobj3.getGaussianBlurredImage(), 'gblurimg' + str(count), 'results/testset/' + str(count) + '/' + 'gblurimg' + str(count) + '.jpg'),
                      (feobj3.getSelectedContourImg(), 'slccntimg' + str(count), 'results/testset/' + str(count) + '/' + 'slccntimg' + str(count) + '.jpg'),
                      (feobj3.getBoundingRectImg(), 'bndrectimg' + str(count), 'results/testset/' + str(count) + '/' + 'bndrectimg' + str(count) + '.jpg'),
                      (feobj3.getBoundedCircImg(), 'bndcircimg' + str(count), 'results/testset/' + str(count) + '/' + 'bndcircimg' + str(count) + '.jpg')])
        __showHaralickFeatures(feobj)
        __showTamuraFeatures(feobj2)
        __showKingsFeatures(feobj4)
        __showGaborPhysicalFeatures(feobj3)
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
        if(str(input('Do you want to enter more images?? \n')) == 'y'):
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

def __predictFromSavedTestCase():
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

def __printPredResWithProperFormatting(predres, type=None):
    if (type == 'SVM'):
        print("FOR SVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['SVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['SVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'SVR'):
        print("FOR SVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['SVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['SVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'NuSVM'):
        print("FOR NuSVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['NuSVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['NuSVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['NuSVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'NuSVR'):
        print("FOR NuSVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['NuSVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['NuSVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['NuSVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'LinSVM'):
        print("FOR LinSVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['LinSVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['LinSVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['LinSVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'LinSVR'):
        print("FOR LinSVR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['LinSVR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['LinSVR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['LinSVR'])['Accuracy'] * 100) + "\n")
    elif (type == 'MLPC'):
        print("FOR MLPC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['MLPC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['MLPC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['MLPC'])['Accuracy'] * 100) + "\n")
    elif (type == 'MLPR'):
        print("FOR MLPR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['MLPR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['MLPR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['MLPR'])['Accuracy'] * 100) + "\n")
    elif (type == 'DTC'):
        print("FOR DTC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['DTC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['DTC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['DTC'])['Accuracy'] * 100) + "\n")
    elif (type == 'DTR'):
        print("FOR DTR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['DTR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['DTR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['DTR'])['Accuracy'] * 100) + "\n")
    elif (type == 'RFC'):
        print("FOR RFC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['RFC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['RFC'])['Accuracy'] * 100) + "\n")
    elif (type == 'RFR'):
        print("FOR RFR - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFR'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['RFR'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['RFR'])['Accuracy'] * 100) + "\n")
    else:
        print("Please enter the correct model acronym!! Your imaginary model does not exist in our model dictionary, sorry!! \n")
        print("Now exiting from prediction mode!! \n")

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
     return list([str(flnm) for flnm in os.listdir(loc) if os.path.isfile(os.path.join(loc, flnm))])

def main_menu():
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^_______WELCOME TO THE MELANOMA-PREDICTION PROGRAM_______^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n")
    print("\t This is a nascent approach towards detecting Melanoma-Skin-Lesion, using OpenCV, NumPY, Matplotlib and SciKit in Python Programming Language. \n")
    print("\t This project utilizes some of the core concepts of \'DIGITAL IMAGE PROCESSING\' & \'MACHINE LEARNING\'. \n")
    print("\t This program can categorize the cancerous-lesion as Malignant, Benign or Negative. \n")
    print("\t Try understanding the meaning of each option, before selecting the appropriate one. \n")
    while (True):
       print("\t Available options are given below : \n")
       print("\t 1.Create \'training-dataset\' from the images of known ->MELANOMA<- types!! \n")
       print("\t 2.Train classifiers and regressors on the created \'training-dataset\'!! \n")
       print("\t 3.Create \'testing-dataset\' from the supervised images in temp folder!! \n")
       print("\t 4.Predict results from the \'testcase.npz\' numpy file!! \n")
       print("\t 5.Print \'feature-descriptors\' of images strored in numpy files, \'dataset.npz\' or \'testcase.npz\'!! \n")
       print("\t 6.Plot \'Classifier/Regressor\' graphs!! \n")
       print("\t 7.Add the \'feature-sets\' to \'testcase.npz\' or \'dataset.npz\' numpy files, to make mlmodels more accurate!! \n")
       print("\t 8.Print only the selected \'feature-sets\' of an image!! \n")
       print("\t 9.List files present in valid \'project-directories\'!! \n")
       print("\t 10.Get color plates of an image!! \n")
       print("\t Enter \'e\' to exit!! \n")
       c = str(input("Enter your choice - \n"))
       if (c == '1'):
           print("If you see a \'results\' folder in the root directory of the project, delete the \'dataset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your corresponding images in the \'images\' folder under the \'malignant\', \'benign\' or \'negative\' directories. \n")
           print("If you haven't already made the directories, please make them and place the corresponding images in them. \n")
           print("The image file-names must be numeric starting from 0 in sequence under each category folder. \n")
           print("Eg. - 0.jpg, 1.jpg, 2.jpg, ..... etc \n")
           print("You must provide images under each category!!! \n")
           input("Just press any key when your are ready : \n")
           __createDataSet("malignant", int(input("Enter the number of images you placed under the \'images/malignant\' directory - \n")))
           __createDataSet("benign", int(input("Enter the number of images you placed under the \'images/benign\' directory - \n")))
           __createDataSet("negative", int(input("Enter the number of images you placed under the \'images/negative\' directory - \n")))
           print("\'Training-Dataset\' successfully generated!! \n")
           print("This dataset consists of the features-array of the corresponding images and their classified types. \n")
           print("All results are stored in the numpy file \'dataset.npz\'. \n")
           print("Total training-images count : %d \n" % imgcount)
       elif (c == '2'):
           print("Now we'll train our various classifiers and regressors on the training-data stored in the \'dataset.npz\' numpy file. \n")
           print("All machine-learning models will be saved as individual \'.pkl\' files in the \'mlmodels\' python-package. \n")
           __createAndTrainMlModels()
           print("Training is now complete!! \n")
       elif (c == '3'):
           print("If you see a \'results\' folder in the root directory of the project, delete the \'testset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your test-images in the \'temp\' folder. \n")
           print("If you haven't already made the directories, please make them and place the test-images in them. \n")
           input("Just press any key when your are ready : \n")
           __getTestImages()
           print("\'Testing-Dataset\' successfully generated!! \n")
           print("This dataset consists of the features-array of the test-images and their supervised-classified types. \n")
           print("All results are stored in the numpy file \'testset.npz\' \n")
       elif (c == '4'):
           print("This will predict results from \'testcase.npz\' and also calculate the prediction accuracy of the individual models. \n")
           pred_res = __predictFromSavedTestCase()
           print("Before we start, here is the reference legend __ \n")
           print("\'1\' : MALIGNANT. \n")
           print("\'0\' : BENIGN. \n")
           print("\'-1\' : NEGATIVE. \n")
           while (True):
              type = str(input('Select Classifier/Regressor acronym : \n'))
              if (type in pred_res):
                    __printPredResWithProperFormatting(pred_res, type)
              else:
                  __printPredResWithProperFormatting(pred_res)
                  break
       elif (c == '5'):
           print("This option prints the features of the images stored in the \'testcase.npz\' numpy file, by default. \n")
           print("You can also print the values stored in the \'dataset.npz\' file, just enter this file-name below. \n")
           print("Before you print the feature-contents, make sure that you had previously generated the \'dataset.npz\' and \'testcase.npz\' numpy files. \n")
           __printfeatsfromfile(str(input('Enter the filename : \n')))
           print("PRINTING COMPLETE!!! \n")
       elif (c == '6'):
           print("\t Before you proceed, make sure that you had previously generated the \'dataset.npz\' numpy file which is existing in the root directory of the project!!! \n")
           dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
           print("\t Given below are the set of features, along with their corresponding indexes. \n")
           for count in range(0, featnames.size, 1):
               print(str(count)+". "+str(featnames[count])+" \n")
           print("You have to select a combination of any two features!! \n")
           print("You can also enter multiple combinations, in such a case they will be appended to a list!! \n")
           flist = []
           fnlist = []
           while(True):
               feat_corrs = [int(input('Enter the index of first feature ... \n')), int(input('Enter the index of second feature ... \n'))]
               flist.append(feat_corrs)
               fnlist.append([featnames[feat_corrs[0]], featnames[feat_corrs[1]]])
               if (str(input('Do you want to add more feature combinations?? - (y/n) \n')) == 'y'):
                   continue
               else:
                   break
           DSP.plotForAll(dset['featureset'], __convertTargetTypeToInt(dset['result']), flist, fnlist)
           print("DONE!!! \n")
       elif (c == '7'):
           def __modify_flnm(string, number):
               ret_str = ""
               for char in string:
                   if (char.isalpha()):
                       ret_str = ret_str + char
                   else:
                       break
               return (ret_str + str(number) + ".jpg")
           def __case7_inner(ptr, typ, flnumber):
               os.mkdir("results/dataset/" + typ + "/" + str(flnumber))
               for name in __listFilesInDir("results/testset/" + str(ptr)):
                   copyfile(src="results/testset/" + str(ptr) + "/" + name, dst="results/dataset/" + typ + "/" + str(flnumber) + "/" + __modify_flnm(name, flnumber))
           print("This option creates a modified \'dataset.npz\' numpy file. \n")
           print("This file includes the feature-sets and the supervised classification results of the test-images. \n")
           print("All the employed mlmodels are automatically re-trained iteratively, on the new modified training-dataset at the end of this step. \n")
           print("Please bear in mind, that re-training the models on the same test-set again, will result in errors. \n")
           featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
           nfls = list([len(__listFilesInDir("images/" + str(cls))) for cls in ('benign', 'malignant', 'negative')])
           trainset, testset = (np.load('dataset.npz'))['dset'], (np.load('testcase.npz'))['dset']
           for feat, index in zip(testset, range(0, testset.size, 1)):
               if (feat[1] == 'benign'):
                   copyfile(src="temp/"+str(index)+".jpg", dst="images/"+str(feat[1])+"/"+str(nfls[0])+".jpg")
                   __case7_inner(index, feat[1], nfls[0])
                   trainset = np.insert(trainset, (nfls[1]+nfls[0]), feat, 0)
                   nfls[0] = nfls[0] + 1
               elif (feat[1] == 'malignant'):
                   copyfile(src="temp/"+str(index)+".jpg", dst="images/"+str(feat[1])+"/"+str(nfls[1])+".jpg")
                   __case7_inner(index, feat[1], nfls[1])
                   trainset = np.insert(trainset, nfls[1], feat, 0)
                   nfls[1] = nfls[1] + 1
               elif (feat[1] == 'negative'):
                   copyfile(src="temp/"+str(index)+".jpg", dst="images/"+str(feat[1])+"/"+str(nfls[2])+".jpg")
                   __case7_inner(index, feat[1], nfls[2])
                   trainset = np.insert(trainset, (nfls[1]+nfls[0]+nfls[2]), feat, 0)
                   nfls[2] = nfls[2] + 1
               else:
                   pass
           np.savez('dataset.npz', dset=trainset, featnames=featnames)
           __createAndTrainMlModels()
       elif (c == '8'):
           def __case8_inner(img_gry, img_col):
                print("Options for selecting the feature-sets are as follows : \n")
                print("a. Print \'Haralick-Texture\' features. \n")
                print("b. Print \'Tamura-Texture\' features. \n")
                print("c. Print \'King-Texture\' features. \n")
                print("d. Print \'Gabor\' physical features. \n")
                print("Any other character input, will result in a default case, displaying \'Feature-Set not found!! Sorry!\' \n")
                chc = str(input("Enter your choice : \n"))
                if (chc == 'a'):
                    __showImages([(img_col, 'imgcol', None), (img_gry, 'imggray', None)])
                    __showHaralickFeatures(har.HarFeat(img_gry))
                elif (chc == 'b'):
                    tobj = tam.TamFeat(img_gry)
                    __showImages([(img_col, 'imgcol', None), (img_gry, 'imggray', None), (tobj.getPrewittHorizontalEdgeImg(), 'PrewittX', None), (tobj.getPrewittVerticalEdgeImg(), 'PrewittY', None), (tobj.getCombinedPrewittImg(), 'PrewittIMG', None)])
                    __showTamuraFeatures(tobj)
                elif (chc == 'c'):
                    __showImages([(img_col, 'imgcol', None), (img_gry, 'imggray', None)])
                    __showKingsFeatures(k.KingFeat(img_gry))
                elif (chc == 'd'):
                    gobj = g.Gabor(img_gry, img_col)
                    __showImages([(img_col, 'imgcol', None), (img_gry, 'imggray', None), (gobj.getGaussianBlurredImage(), 'gblurimg', None), (gobj.getSelectedContourImg(), 'slccntimg', None), (gobj.getBoundingRectImg(), 'bndrectimg', None), (gobj.getBoundedCircImg(), 'bndcircimg', None)])
                    __showGaborPhysicalFeatures(gobj)
                else:
                    print("Oopsy-Daisy!! Feature-Set not found!! Sorry!! Please enter the correct character!! \n")
           print("\t In this step we'll get the selected feature-sets of the input-image(will be converted to gray-scale) and print them on screen!! \n")
           print("\t Initially we'll perform some pre-processing operations on the original gray-scale image and create some variants!! \n")
           print("\t You'll get the option of selecting either the pre-processed image variants or the original gray-scale image for getting the selected feature-set!! \n")
           print("\t Before you proceed, make-sure to create a \'test\' directory inside the project root and place the required images there!! \n")
           print("\t No-worries if you had created the \'test\' directory before, just place your images of choice in there!! \n")
           print("\t All features are generated over gray-scale images, hence your original color image will be converted to it's corresponding gray-scale image!! \n")
           obj = p.Prep('test/' + str(input("Enter file-name of image : \n")))
           print("Otsu's threshold-level for the input-image is %d \n" % obj.getOtsuThresholdLevel())
           print("Options for selecting the image-variant are as follows : \n")
           print("a. Select inverted gray-scale image. \n")
           print("b. Select segmented binary image. \n")
           print("c. Select segmented gray-scale image. \n")
           print("Any other character input, will result in a default case, where the selected image will be the original gray-scale image. \n")
           chc = str(input("Enter your choice : \n"))
           if (chc == 'a'):
               __case8_inner(obj.getInvrtGrayImg(), obj.getActImg())
           elif (chc == 'b'):
               __case8_inner(obj.getBinaryImg(), obj.getSegColImg())
           elif (chc == 'c'):
               __case8_inner(obj.getSegGrayImg(), obj.getSegColImg())
           else:
               __case8_inner(obj.getGrayImg(), obj.getActImg())
       elif (c == '9'):
           print("\t Before you use this option, it is recommended that you go through the project-structure once. \n")
           for fls in __listFilesInDir(str(input("\t Please Enter A Valid Directory Path. \n"))):
               print(fls + '\n')
           print("\t Successfully Listed All File-Names. \n")
       elif (c == '10'):
           print("\t This option displays the 3 individual color-plates(R-G-B) of the test image placed in the \'/test\' directory. \n")
           print("\t Before you proceed make sure that you have created the \'test\' directory and placed the corresponding images there. \n")
           obj = p.Prep('test/' + str(input("Enter file-name of image : \n")))
           __showImages([(obj.getActImg(), 'act_img', None), (obj.getSegColImg(), 'act_seg_img', None), (obj.getColorPlates(obj.getActImg(), 'R'), 'act_img_red', None), (obj.getColorPlates(obj.getActImg(), 'G'), 'act_img_green', None), (obj.getColorPlates(obj.getActImg(), 'B'), 'act_img_blue', None), (obj.getColorPlates(obj.getSegColImg(), 'R'), 'seg_img_red', None), (obj.getColorPlates(obj.getSegColImg(), 'G'), 'seg_img_green', None), (obj.getColorPlates(obj.getSegColImg(), 'B'), 'seg_img_blue', None)])
           print("\t DONE !!! \n")
       else:
           print("Thank-You For Using This Program!!!")
           print("Now Exiting.")
           break

main_menu()



