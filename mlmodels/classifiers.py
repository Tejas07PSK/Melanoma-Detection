import numpy as np
from sklearn.svm import SVC

def useSVMClassifier(feat)
    import numpy as np
    from sklearn.svm import SVC
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    clf = SVC()
    clf.fit(X, y)