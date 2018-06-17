import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import clone
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

(__no_of_clas, __clas_types, __col_map, __plt_interval, __plt_interval_pp, __rnd_seed) = (3, ('benign', 'malignant', 'negative'), plt.cm.RdYlGn, 0.05, 0.5, 13)
cloned_classifiers = [joblib.load('mlmodels/Mel_SVM.pkl'), joblib.load('mlmodels/Mel_NuSVM.pkl'), joblib.load('mlmodels/Mel_LinSVM.pkl'), joblib.load('mlmodels/Mel_MLPC.pkl'), joblib.load('mlmodels/Mel_DTC.pkl'), joblib.load('mlmodels/Mel_RFC.pkl')]

def plotForAll(X, Y, ftup, feats):
    titles = ('SVM', 'NuSVM', 'LinSVM', 'MLPC', 'DTC', 'RFC')
    index = np.arange(0, X.shape[0], 1)
    for idx_pair, feat in zip(ftup, feats):
            x = X[:, idx_pair]
            np.random.seed(__rnd_seed)
            np.random.shuffle(index)
            x = x[index]
            y = Y[index]
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            cor_tup = plt.subplots(len(ftup), len(cloned_classifiers))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            for i in range(0, ((cor_tup[1]).shape)[0], 1):
                for mdl, title, j in zip(cloned_classifiers, titles, range(0, ((cor_tup[1]).shape)[1], 1)):
                    clf = (clone(mdl)).fit(x, y)
                    scr = clf.score(x, y)
                    print(scr)
                    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, ((x_max - x_min) / 100.0)), np.arange(y_min, y_max, ((y_max - y_min) / 100.0)))
                    if isinstance(mdl, RandomForestClassifier):
                        alpha_blend = 1.0 / len(mdl.estimators_)
                        for tree in mdl.estimators_:
                            ((cor_tup[1])[i, j]).contourf(xx, yy, (tree.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape), alpha=alpha_blend, cmap=__col_map)
                    else:
                        ((cor_tup[1])[i, j]).contourf(xx, yy, (mdl.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape), cmap=__col_map)
                    xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, ((x_max - x_min) / 50.0)), np.arange(y_min, y_max, ((y_max - y_min) / 50.0)))
                    ((cor_tup[1])[i, j]).scatter(xx_coarser, yy_coarser, s=15, c=mdl.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape), cmap=__col_map, edgecolors="none")
                    ((cor_tup[1])[i, j]).scatter(x[:, 0], x[:, 1], c=y, cmap=__col_map, edgecolor='k', s=20)
                    ((cor_tup[1])[i, j]).set_xlim(xx.min(), xx.max())
                    ((cor_tup[1])[i, j]).set_ylim(yy.min(), yy.max())
                    ((cor_tup[1])[i, j]).set_xlabel(feat[0])
                    ((cor_tup[1])[i, j]).set_ylabel(feat[1])
                    ((cor_tup[1])[i, j]).set_xticks(())
                    ((cor_tup[1])[i, j]).set_yticks(())
                    ((cor_tup[1])[i, j]).set_title(title)
    plt.suptitle("Plot of Classifiers on feature subsets of the Melanoma-Dataset")
    plt.show()