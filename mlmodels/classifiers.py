import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier as DTC

class Classifiers(object):

    def __init__(self):
        self.__svm_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        self.__nusvm_clf = NuSVC(cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        self.__linsvm_clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
print(list(dset['featureset']))
print(list(dset['result']))
X = np.array(list(dset['featureset']))
Y = np.array(list(dset['result']))
"""clf = SVC()
clf.fit(X, Y)
clf = MLP()
print(clf)
clf.fit(X,Y)
joblib.dump(clf, 'Mel_MLP.pkl')
print(clf.predict(list(dset['featureset'])))
joblib.dump(clf, 'Mel_SVM.pkl')
clf2 = joblib.load('Mel_SVM.pkl')
print(clf2)"""

clf = DTC()
clf = clf.fit(X, Y)
print(clf.predict(X))
print(clf)
joblib.dump(clf, 'Mel_DTC.pkl')