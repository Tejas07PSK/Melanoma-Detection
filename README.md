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
        []  _Official 'matplotlib' docs_ :point_right: [matplotlib.org/docs](https://matplotlib.org/contents.html).  
        (**Note.-** For installing _MatPlotLib_ through pip, type `python -m pip --user install matplotlib`.)  
    + **'SciPy'**.  
        []  _About 'scipy'_ :point_right: [wikipedia.org/scipy](https://en.wikipedia.org/wiki/SciPy).  
        []  _Official 'scipy' documentations_ :point_right: [scipy.org/docs](https://www.scipy.org/docs.html).  
        (**Note.-** For installing _SciPy_ through pip, type `python -m pip --user install scipy`.)  
    + **'OpenCV'**.  
        []  _About 'opencv'_ :point_right: [wikipedia.org/opencv](https://en.wikipedia.org/wiki/OpenCV).  
        []  _Official 'opencv-python' online tutorial_ :point_right: [opencv-python.org/tutorials](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).  
        []  _Official 'opencv-python' documentation_ :point_right: [docs.opencv.org/python](https://docs.opencv.org/3.0-beta/index.html).  
        (**Note.-** For installing _OpenCV_ through pip, type `python -m pip --user install opencv-python`.)  
    + **'Scikit-Learn'**.  
        []  _About 'scikit-learn'_ :point_right: [wikipedia.org/Scikit-Learn](https://en.wikipedia.org/wiki/Scikit-learn).  
        []  _Official 'scikit-learn' documentation_ :point_right: [scikit-learn.org/docs](http://scikit-learn.org/stable/documentation.html).  
        (**Note.-** For installing _Scikit-Learn_ through pip, type `python -m pip --user install sklearn`.)  
        
### _Running the Application_ :
* Before you download this application, make sure you have installed 'Python3' along with the dependency modules.  
* If you are using any integrated development environment like 'PyCharm', you can simply clone this repository to your  
  project directory, using the git-commandline tools, just type `git clone https://github.com/Tejas07PSK/Melanoma-Detection.git`.  
* As this a 'console/commandline/terminal' application you can simply download this repository as `Melanoma-Detection.zip`
  compressed file and then extract it accordingly. Now, navigate to the extracted folder in terminal  
  and then run this program by typing `python Main.py`.  
* As you run this application, you will be greeted with the following text as shown in the screenshot.  
  ![Screenshot-1](https://user-images.githubusercontent.com/29245625/43370268-8dd88c00-9399-11e8-9819-803ecfe350ce.png)  
* Now, if you select option **_'1'_**, you will go through the following phases as shown in the screenshots.  
  ![Screenshot-2](https://user-images.githubusercontent.com/29245625/43410171-9edabb4a-9443-11e8-8fa9-d03b129d8b70.png)  
  ![Screenshot-3](https://user-images.githubusercontent.com/29245625/43410312-19338d36-9444-11e8-9acc-d9802badbc8e.png)  
  ![Screenshot-4](https://user-images.githubusercontent.com/29245625/43415670-d6bec4b0-9453-11e8-845e-e2b76d368d7d.png)  
  ![Screenshot-5](https://user-images.githubusercontent.com/29245625/43416632-7c6ea3c4-9456-11e8-8f16-012a83780e21.png)  
  ![Screenshot-6](https://user-images.githubusercontent.com/29245625/43417829-b8a87a10-9459-11e8-8bb1-ede7aa7efb6f.png)  
  ![Screenshot-7](https://user-images.githubusercontent.com/29245625/43418479-a267ae54-945b-11e8-9f6f-833ebf3b4422.png)  
  ![Screenshot-8](https://user-images.githubusercontent.com/29245625/43418558-e209842e-945b-11e8-952f-bfd21c9d9b3f.png)  
* Next, if you select option **_'2'_**, you will get the following output as shown in the screenshot.  
  ![Screenshot-9](https://user-images.githubusercontent.com/29245625/43418835-d82d5ccc-945c-11e8-9654-5236dd32e7b2.png)  
* Next, if you select option **_'3'_**, the operation phase will be very similar to option **_'1'**, as shown in the following screenshots.  
  ![Screenshot-10](https://user-images.githubusercontent.com/29245625/43420229-caf2edd4-9460-11e8-9529-d555f554e60e.png)  
  ![Screenshot-11](https://user-images.githubusercontent.com/29245625/43420753-643ee6ea-9462-11e8-986e-5327a462db2c.png)  
  ![Screenshot-12](https://user-images.githubusercontent.com/29245625/43421789-63163c66-9465-11e8-8542-57a60765b055.png)  
* Next, if you select option **_'7'_**, the existing ml-models will be iteratively trained with the new test-images  
  placed in the _'/temp'_ folder, look at this screenshot.  
  ![Screenshot-13](https://user-images.githubusercontent.com/29245625/43423442-1552a118-946a-11e8-9de1-8f3565544a48.png)  
* Now, if you select option **_'5'_**, you will get the following output, as shown in the screenshots.  
  ![Screenshot-14](https://user-images.githubusercontent.com/29245625/43424125-fd8b8782-946b-11e8-99d3-71e5c2c5531e.png)  
  ![Screenshot-15](https://user-images.githubusercontent.com/29245625/43424211-357f7126-946c-11e8-8d49-d9761016c063.png)  
* If you select option **_'8'_**, you'll get the following output as shown in the screenshots.  
  ![Screenshot-16](https://user-images.githubusercontent.com/29245625/43424458-2638c41e-946d-11e8-9557-f693a9a1c355.png)  
  ![Screenshot-17](https://user-images.githubusercontent.com/29245625/43424749-02550886-946e-11e8-9901-a996376f569c.png)  
  ![Screenshot-18](https://user-images.githubusercontent.com/29245625/43424807-26170bb6-946e-11e8-92d8-477912d0b5c3.png)  
* Option **_'9'_** is used for getting a list of files present in any valid project directory, look at the screenshot.  
  ![Screenshot-19](https://user-images.githubusercontent.com/29245625/43424995-d9a94a18-946e-11e8-8906-5fcfb9dba059.png)  
* Option **_'6'_** is used for plotting the decision-surfaces of the various classifiers/regressors as shown below in the screeenshots.  
  ![Screenshot-20](https://user-images.githubusercontent.com/29245625/43425489-62e3dd7e-9470-11e8-8dde-563a697cefd4.png)  
  ![Screenshot-21](https://user-images.githubusercontent.com/29245625/43425542-88789eda-9470-11e8-87b9-70877ec195e7.png)  
  ![Screenshot-22](https://user-images.githubusercontent.com/29245625/43425571-a94d18d4-9470-11e8-8e1d-a2724f23897c.png)  
  ![Screenshot-23](https://user-images.githubusercontent.com/29245625/43425571-a94d18d4-9470-11e8-8e1d-a2724f23897c.png)  
  ![Screenshot-24](https://user-images.githubusercontent.com/29245625/43425636-d8a888f2-9470-11e8-8ed8-5dcb9c25e6a9.png)  
  ![Screenshot-25](https://user-images.githubusercontent.com/29245625/43425662-f2a74950-9470-11e8-89ae-5bbe57575d1c.png)  
