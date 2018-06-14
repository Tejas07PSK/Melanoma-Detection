import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import clone
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

(__no_of_clas, __clas_types, __col_map, __plt_interval, __plt_interval_pp, __rnd_seed) = (3, ('benign', 'malignant', 'negative'), plt.cm.RdYlGn, 0.05, 0.5, 13)
cloned_classifiers = [clone(joblib.load('mlmodels/Mel_SVM.pkl')), clone(joblib.load('mlmodels/Mel_NuSVM.pkl')), clone(joblib.load('mlmodels/Mel_LinSVM.pkl')), clone(joblib.load('mlmodels/Mel_MLPC.pkl')), clone(joblib.load('mlmodels/Mel_DTC.pkl')), clone(joblib.load('mlmodels/Mel_RFC.pkl'))]

def plotForAll(X, Y, ftup):
    for idx_pair in ftup:
        for mdl in cloned_classifiers:
            x = X[:, idx_pair]
            index = np.arange(X.shape[0])
            np.random.seed(__rnd_seed)
            np.random.shuffle(index)
            x = x[index]
            y = Y[index]
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            clf = mdl.fit(x, y)
            scr = clf.score(X, y)
