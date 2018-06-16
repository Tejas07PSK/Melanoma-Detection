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
cloned_classifiers = [joblib.load('mlmodels/Mel_SVM.pkl'), joblib.load('mlmodels/Mel_NuSVM.pkl'), joblib.load('mlmodels/Mel_LinSVM.pkl'), joblib.load('mlmodels/Mel_MLPC.pkl'), joblib.load('mlmodels/Mel_DTC.pkl'), joblib.load('mlmodels/Mel_RFC.pkl')]

def plotForAll(X, Y, ftup):
    plot_idx = 1
    titles = ('SVM', 'NuSVM', 'LinSVM', 'MLPC')
    index = np.arange(X.shape[0])
    for idx_pair in ftup:
        for mdl in cloned_classifiers[0:5]:
            x = X[:, idx_pair]
            np.random.seed(__rnd_seed)
            np.random.shuffle(index)
            x = x[index]
            y = Y[index]
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            clf = mdl.fit(x, y)
            scr = clf.score(X, y)
            model_title = str(type(mdl)).split(".")[-1][:-2][:-len("Classifier")]
            plt.subplot(len(ftup), len(cloned_classifiers), plot_idx)
            if (plot_idx <= len(mdl)):
                plt.title(model_title)
            x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, __plt_interval), np.arange(y_min, y_max, __plt_interval))
