import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.tree import DecisionTreeClassifier as DTC

class Classifiers(object):

    def __init__(self):
        self.__svm_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        self.__nusvm_clf = NuSVC(cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        self.__linsvm_clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
        self.__mlpc_clf = MLPC(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
        self.__dtc_clf = DTC(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
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