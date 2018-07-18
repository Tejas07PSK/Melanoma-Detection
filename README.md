# MELANOMA-DETECTION
###### What is Melanoma?
**_'Melanoma'_**, also known as **_'Malignant_Melanoma'_**, is a type
of cancer that develops from the pigment-containing cells known
as _melanocytes_. Melanomas typically occur on the skin, but may rarely
occur in the mouth, intestines, or eye. The primary cause of
melanoma is ultraviolet light (UV) exposure in those with low levels of
skin pigment. The UV light may be from either the sun or from other
sources, such as tanning devices. About 25% develop from moles.  
**Read more at :->[wikipedia.org/melanoma](https://en.wikipedia.org/wiki/Melanoma)**

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
modular console program, written in Python3. Anyone, with some basic
programming knowledge will be able to run this app easily.  
Simply, this console app tries to predict the nature of a 'skin-lesion'
image, served as an input.  
To keep things simple, we have trained our machine-learning models, to
classify the input image as one of the three types:  
   + **NEGATIVE** - Represents a skin-lesion that is not melanoma._(-1)_
   + **BENIGN** - Represents a skin-lesion that is an early-stage melanoma._(0)_
   + **MALIGNANT** - Represents a skin-lesion that is highly cancerous._(1)_   

The application consists of five core modules, namely:  
   1. _Main.py_  (Driver module for the entire application).  
   2. **featext**  ('quatified-features' extraction module for the input_image).  
      + **physical**  ('physical-features' extraction sub-module for the input_image).  
        - _Gabor.py_  (Extracts "Gabor's" physical-features from the input_image).  
      + **texture**   ('textural-features' extraction module for the input_image).  
        - _Haralick.py_  (Extracts "Haralick's" texture-features from the input_image).  
        - _King.py_  (Extracts "King's" texture-features from the input_image).  
        - _Tamura.py_  (Extracts "Tamura's" texture-features from the input_image).  
   3. **mlmodels**  (input_image classification/regression module).  
      + _Classifiers.py_  (Predicts the class of the input_image).  
      + _DecisionSurfacePlotter.py_  (Plots the decision surfaces of the various classifiers/regressors, based on the selected features).  
      + _Mel_DTC.pkl_  (Persistently stores the trained 'Decision Tree Classifier' object).  
      + _Mel_DTR.pkl_  (Persistently stores the trained 'Decision Tree Regressor' object).  
      + _Mel_LinSVM.pkl_  (Persistently stores the trained 'Linear-Support Vector Machine Classifier' object).  
      + _Mel_LinSVR.pkl_  (Persistently stores the trained 'Linear-Support Vector Machine Regressor' object).  
      + _Mel_MLPC.pkl_  (Persistently stores the trained 'Multi-Layer Perceptron Classifier' object).  
      + _Mel_MLPR.pkl_  (Persistently stores the trained 'Multi-Layer Perceptron Regressor' object).  
      + _Mel_NuSVM.pkl_  (Persistently stores the trained 'Nu-Support Vector Machine Classifier' object).  
      + _Mel_NuSVR.pkl_  (Persistently stores the trained 'Nu-Support Vector Machine Regressor' object).   
      + _Mel_RFC.pkl_  (Persistently stores the trained 'Random Forest Classifier' object).  
      + _Mel_RFR.pkl_  (Persistently stores the trained 'Random Forest Regressor' object).  
      + _Mel_SVM.pkl_  (Persistently stores the trained 'Support Vector Machine Classifier' object).  
      + _Mel_SVR.pkl_  (Persistently stores the trained 'Support Vector Machine Regressor' object).  
   4. **preprocessing**  (input_image preprocessing module).  
      + _Prep.py_  (Performs generic image pre-processing operations).  
   5. **util**  (General library utility module).  
      + _Util.py_  (Performs routine data-structural operations ... insertion, searching, sorting etc).  

## How the application works?  
This application works according to the following folds :  
1.  Firstly, a 'training-set' data is generated from a collection of various skin-lesion images placed in their respective  
    class folders i.e., _'images/benign'_, _'images/malignant'_, _'images/negative'_. These images are pre-processed and  
    a set of quantified-features are extracted from them, which comprises the 'training-set' data.  
2.  Next, the above generated training data, is then passed on to the various classifier/regressor objects for training/learning.  
3.  Now, the trained models are persistently saved as python objects or pickle units, in individual '.pkl' files.  
4.  Now, a set of input_images in need of classification are placed in the _'temp'_ folder.
5.  Next, the program takes each input_image pre-processes it and extracts the necessary features from it.  
6.  The features generated from the pre-processed input_images are then passed on to the various machine-learning models,  
    which in turn predicts the nature of each input_image accordingly.  
7.  Since, the learning process here is supervised, a 'prediction-accuracy' is generated for each model.  
8.  Finally, the results from the model with the highest 'prediction-accuracy' are selected.  



        
         
        
      
    
