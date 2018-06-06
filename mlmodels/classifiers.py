import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier as DTC

class Classifiers:

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