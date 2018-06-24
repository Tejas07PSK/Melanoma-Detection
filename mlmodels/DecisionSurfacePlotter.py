import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Pchs
from sklearn import clone
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

(__col_map, __rnd_seed) = (plt.cm.RdYlGn, 13)

def plotForAll(X, Y, ftup, feats):
    classifiers = [joblib.load('mlmodels/Mel_SVM.pkl'), joblib.load('mlmodels/Mel_NuSVM.pkl'), joblib.load('mlmodels/Mel_LinSVM.pkl'), joblib.load('mlmodels/Mel_MLPC.pkl'), joblib.load('mlmodels/Mel_DTC.pkl'), joblib.load('mlmodels/Mel_RFC.pkl')]
    regressors = [joblib.load('mlmodels/Mel_SVR.pkl'), joblib.load('mlmodels/Mel_NuSVR.pkl'), joblib.load('mlmodels/Mel_LinSVR.pkl'), joblib.load('mlmodels/Mel_MLPR.pkl'), joblib.load('mlmodels/Mel_DTR.pkl'), joblib.load('mlmodels/Mel_RFR.pkl')]
    titles = ()
    models = []
    for nplot in range(0, 2, 1):
        if (nplot == 0):
            print("_-__-_ IN CASE OF CLASSIFERS _-__-_ \n")
            titles = ('SVM', 'NuSVM', 'LinSVM', 'MLPC', 'DTC', 'RFC')
            plt.figure("Decision Surface For Classifiers", edgecolor='b')
            plt.suptitle("Plot of Classifiers on feature subsets of the Melanoma-Dataset")
            models = classifiers
        else:
            print("_-__-_ IN CASE OF REGRESSORS _-__-_ \n")
            titles = ('SVR', 'NuSVR', 'LinSVR', 'MLPR', 'DTR', 'RFR')
            plt.figure("Decision Surface For Regressors", edgecolor='b')
            plt.suptitle("Plot of Regressors on feature subsets of the Melanoma-Dataset")
            models = regressors
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
                print(" " + feat[0] + " Vs. " + feat[1] + " :- \n")
                for mdl, title in zip(models, titles):
                        obj = plt.subplot(len(ftup), len(models), plot_index)
                        clf = (clone(mdl)).fit(x, y)
                        scr = clf.score(x, y)
                        print("Feasibility Score For " + title + " Model - " + str(scr * 100))
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
                        (obj).set_xlim(xx.min(), xx.max())
                        (obj).set_ylim(yy.min(), yy.max())
                        (obj).set_xlabel(feat[0])
                        (obj).set_ylabel(feat[1])
                        (obj).set_xticks(())
                        (obj).set_yticks(())
                        if (plot_index == (len(ftup) * len(models))):
                            plt.legend(handles=[Pchs.Patch(color='red', label='MALIGNANT'), Pchs.Patch(color='yellow', label='BENIGN'), Pchs.Patch(color='green', label='NEGATIVE')], loc='upper right', fontsize='small')
                        if (plot_index <= len(models)):
                            (obj).set_title(title)
                        plot_index += 1
                print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: \n \n")
    plt.show()