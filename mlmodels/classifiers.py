import numpy as np
from sklearn.svm import SVC

X = np.array([[-0.000654321, -1.56712], [23455567896.33, -1.9877], [1124.22, 1.66701], [29999, -0]])
Y = np.array(['Cat', 'Dog', 'Tiger', 'Bear'])
clf = SVC()
clf.fit(X, Y)
print(clf.predict([[1097.01, 0.9987]]))