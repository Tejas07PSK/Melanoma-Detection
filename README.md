# MELANOMA-DETECTION
###### What is Melanoma?
**_'Melanoma'_**, also known as **_'Malignant_Melanoma'_**, is a type
of cancer that develops from the pigment-containing cells known
as _melanocytes_. Melanomas typically occur on the skin, but may rarely
occur in the mouth, intestines, or eye. The primary cause of
melanoma is ultraviolet light (UV) exposure in those with low levels of
skin pigment. The UV light may be from either the sun or from other
sources, such as tanning devices. About 25% develop from moles.  
**Read More At :->[wikipedia.org/melanoma](https://en.wikipedia.org/wiki/Melanoma)**

## About The Repository - 
This repo holds the source code for the Melanoma-Detection Application.
Given below is the _'Project Structure'_ :  
  
    .
    |   Main.py
    |   dataset.npz
    |   testcase.npz
    |   README.md
    |---featext
    |     |---physical
    |     |     |   __init__.py
    |     |     |   Gabor.py
    |     |---texture
    |     |     |   __init__.py
    |     |     |   Haralick.py
    |     |     |   King.py
    |     |     |   Tamura.py
    |     |   __init__.py
    |---images
    |     |---benign
    |     |     |   'img_number'.jpg
    |     |---malignant
    |     |     |   'img_number'.jpg
    |     |---negative
    |     |     |   'img_number'.jpg
    |---mlmodels
    |     |   Classifiers.py
    |     |   DecisionSurfacePlotter.py
    |     |   Mel_DTC.pkl
    |     |   Mel_DTR.pkl
    |     |   Mel_LinSVM.pkl
    |     |   Mel_LinSVR.pkl
    |     |   Mel_MLPC.pkl
    |     |   Mel_MLPR.pkl
    |     |   Mel_NuSVM.pkl
    |     |   Mel_NuSVR.pkl
    |     |   Mel_RFC.pkl
    |     |   Mel_RFR.pkl
    |     |   Mel_SVM.pkl
    |     |   Mel_SVR.pkl
    |     |   __init__.py
    |---preprocessing
    |     |   Prep.py
    |     |   __init__.py
    |---results
    |     |---dataset
    |     |     |---benign
    |     |     |     |---'numbers'
    |     |     |     |     |   'images'.jpg
    |     |     |---malignant
    |     |     |     |---'numbers'
    |     |     |     |     |    'images'.jpg
    |     |     |---negative
    |     |     |     |---'numbers'
    |     |     |     |     |    'images'.jpg
    |     |---testset
    |     |     |---benign
    |     |     |     |---'numbers'
    |     |     |     |     |   'images'.jpg
    |     |     |---malignant
    |     |     |     |---'numbers'
    |     |     |     |     |    'images'.jpg
    |     |     |---negative
    |     |     |     |---'numbers'
    |     |     |     |     |    'images'.jpg
    |---temp
    |     |   'img_number'.jpg
    |---test
    |     |   'img_number'.jpg
    |---util
          |   Util.py
          |   __init__.py

## About The Application -
This application does not contain any fancy _UI_, as it is basically a
modular console program, written in Python3.Anyone, with some basic
programming knowledge will be able to run this app easily.  
Simply, this console app tries to predict the nature of a 'skin-lesion'
image, served as an input.  
To keep things simple, we have trained our machine-learning models, to
classify the input image as one of the three types:  
   + **NEGATIVE** - Represents a skin-lesion that is not melanoma._(-1)_
   + **BENIGN** - Represents a skin-lesion that is an early-stage melanoma._(0)_
   + **MALIGNANT** - Represents a skin-lesion that is highly cancerous._(1)_ 
         
     
    
