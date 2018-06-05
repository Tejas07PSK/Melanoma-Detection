import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
print(list(dset['featureset']))
print(list(dset['result']))
X = np.array(list(dset['featureset']))
Y = np.array(list(dset['result']))
clf = SVC()
clf.fit(X, Y)
joblib.dump(clf, 'Mel_SVM.pkl')
clf2 = joblib.load('Mel_SVM.pkl')
print(clf2)
#print(clf.predict([[1097.01, 0.9987]]))