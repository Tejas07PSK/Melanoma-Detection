# MELANOMA-DETECTION
###### What is Melanoma?
**_'Melanoma'_**, also known as **_'Malignant_Melanoma'_**, is a type
of cancer that develops from the pigment-containing cells known
as _melanocytes_. Melanomas typically occur on the skin, but may rarely
occur in the mouth, intestines, or eye. The primary cause of
melanoma is ultraviolet light (UV) exposure in those with low levels of
skin pigment. The UV light may be from either the sun or from other
sources, such as tanning devices. About 25% develop from moles.  
**Read more at :-> [wikipedia.org/melanoma](https://en.wikipedia.org/wiki/Melanoma)**

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
4.  Next, a set of input_images in need of classification are placed in the _'temp'_ folder.
5.  Next, the program takes each input_image pre-processes it and extracts the necessary features from it.  
6.  The features generated from the pre-processed input_images are then passed on to the various  
    machine-learning models,which in turn predicts the nature of each input_image accordingly.  
7.  Since, the learning process here is supervised, a 'prediction-accuracy' is generated for each model.  
8.  Finally, the results from the model with the highest 'prediction-accuracy' are selected.  

## Usage guide -  
### _Pre-requisites_ :  
1. **Python3** ;  
       []  _About 'Python3'_ :point_right: [wikipedia.org/PythonProgrammingLanguage](https://en.wikipedia.org/wiki/Python_(programming_language)).  
       []  _How to install 'Python3'?_ :point_right: [python.org/BeginnersGuide](https://wiki.python.org/moin/BeginnersGuide/Download).  
       []  _Official 'Python3' documentation_ :point_right: [docs.python.org/Python3](https://docs.python.org/3/).  
       []  _GET 'Python3'_ :point_right: [python.org/downloads](https://www.python.org/downloads/).

2. **Python Package Manager** (any one of the below applications will suffice) ;  
    + **pip** :point_right: comes along with the executable 'python-installation' package.  
        []  _About 'pip'_ >>> [wikipedia.org/pip_packagemanager](https://en.wikipedia.org/wiki/Pip_(package_manager)).  
        []  _How install packages using 'pip'?_ >>> [docs.python.org/inastallingPythonPackagesUsingPip](https://docs.python.org/3/installing/index.html).  
    + **anaconda** :point_right: [**anaconda.com/downloads**](https://www.anaconda.com/download/).  
        []  _About 'anaconda'_ >>> [wikipedia.org/conda](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)).  
        []  _How to install 'anaconda'?_ >>> [docs.anaconda.com/installingconda](https://enterprise-docs.anaconda.com/en/latest/install/index.html).  
        []  _How to install 'python-packages' with 'conda'?_ >>> [docs.anaconda.com/packages](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/packages/index.html)  
        []  _Official 'anaconda' documentation?_ >>> [docs.anaconda.com/official](https://enterprise-docs.anaconda.com/en/latest/).  
        
3. **IDE** (optional) ;  
    + **Pycharm** :point_right: [jetbrains.com/getPycharm](https://www.jetbrains.com/pycharm/).  
        []  _About 'Pycharm'_ >>> [wikipedia.org/Pycharm](https://en.wikipedia.org/wiki/PyCharm).  
        []  _How to use 'Pycharm'?_ >>> [jetbrains.com/PycharmGuide](https://www.jetbrains.com/pycharm/documentation/).  

4. **Python Library Dependencies** ;
    + **'NumPy'**.  
        []  _About 'numpy'_ :point_right: [wikipedia.org/numpy](https://en.wikipedia.org/wiki/NumPy).  
        []  _Official 'numpy' manuals_ :point_right: [docs.scipy.org/numpyManuals](https://docs.scipy.org/doc/numpy/).  
        (**Note.-** For installing _NumPy_ through pip, type `python -m pip --user install numpy`.)  
    + **'MatPlotLib'**.  
        []  _About 'matplotlib'_ :point_right: [wikipedia.org/matplotlib](https://en.wikipedia.org/wiki/Matplotlib).  
        []  _Official 'matplotlib' docs :point_right: [matplotlib.org/docs](https://matplotlib.org/contents.html).  
        (**Note.-** For installing _MatPlotLib_ through pip, type `python -m pip --user install matplotlib`.)  
    + **'SciPy'**.  
        []  _About 'scipy'_ :point_right: [wikipedia.org/scipy](https://en.wikipedia.org/wiki/SciPy).  
        []  _Official 'scipy' documentations_ :point_right: [scipy.org/docs](https://www.scipy.org/docs.html).  
        (**Note.-** For installing _SciPy_ through pip, type `python -m pip --user install scipy`.)  
        
         
        
      
    
