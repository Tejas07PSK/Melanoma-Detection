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
    #plt.suptitle("Plot of Classifiers on feature subsets of the Melanoma-Dataset")
    plt.figure('Classifier')
    plt.suptitle("Plot of Classifiers on feature subsets of the Melanoma-Dataset")
    index = np.arange(0, X.shape[0], 1)
    plot_index = 1
    for idx_pair, feat in zip(ftup, feats):
            x = X[:, idx_pair]
            np.random.seed(__rnd_seed)
            np.random.shuffle(index)
            x = x[index]
            y = Y[index]
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            plt.subplots_adjust(wspace=1.0, hspace=1.0)
            print(x)
            for mdl, title in zip(cloned_classifiers, titles):
                    obj = plt.subplot(len(ftup), len(cloned_classifiers), plot_index)
                    clf = (clone(mdl)).fit(x, y)
                    scr = clf.score(x, y)
                    print(scr)
                    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, ((x_max - x_min) / 100.0)), np.arange(y_min, y_max, ((y_max - y_min) / 100.0)))
                    if isinstance(clf, RandomForestClassifier):
                        alpha_blend = 1.0 / len(clf.estimators_)
                        for tree in clf.estimators_:
                            (obj).contourf(xx, yy, (tree.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape), alpha=alpha_blend, cmap=__col_map)
                    else:
                        (obj).contourf(xx, yy, (clf.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape), cmap=__col_map)
                    xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, ((x_max - x_min) / 50.0)), np.arange(y_min, y_max, ((y_max - y_min) / 50.0)))
                    (obj).scatter(xx_coarser, yy_coarser, s=15, c=clf.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape), cmap=__col_map, edgecolors="none")
                    (obj).scatter(x[:, 0], x[:, 1], c=y, marker='p', cmap=__col_map, edgecolor='k', s=20)
                    print((xx.min(), xx.max()))
                    print((yy.min(), yy.max()))
                    print(xx.max() - xx.min())
                    print(yy.max() - yy.min())
                    (obj).set_xlim(xx.min(), xx.max())
                    (obj).set_ylim(yy.min(), yy.max())
                    (obj).set_xlabel(feat[0])
                    (obj).set_ylabel(feat[1])
                    (obj).set_xticks(())
                    (obj).set_yticks(())
                    if (plot_index <= len(cloned_classifiers)):
                        (obj).set_title(title)
                    plot_index += 1
    plt.show()