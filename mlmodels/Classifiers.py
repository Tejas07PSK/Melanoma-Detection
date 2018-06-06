from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score

class Classifiers(object):

    def __init__(self, featureset=None, target=None, mode='predict', path=''):
        if (mode == 'train'):
            self.__svm_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            self.__nusvm_clf = NuSVC(cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            self.__linsvm_clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
            self.__mlpc_clf = MLPC(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(100, 34), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
            self.__dtc_clf = DTC(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
            self.__rfc_clf = RFC(bootstrap=True, class_weight=None, criterion='gini', max_depth=100, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
            (self.__svm_clf, self.__nusvm_clf, self.__linsvm_clf, self.__mlpc_clf, self.__dtc_clf, self.__rfc_clf) = self.__trainAll(X=list(featureset), Y=list(target))
            self.__saveModelsToFile(path)
        else:
            self.__svm_clf = joblib.load(path + 'Mel_SVM.pkl')
            self.__nusvm_clf = joblib.load(path + 'Mel_NuSVM.pkl')
            self.__linsvm_clf = joblib.load(path + 'Mel_LinSVM.pkl')
            self.__mlpc_clf = joblib.load(path + 'Mel_MLPC.pkl')
            self.__dtc_clf = joblib.load(path + 'Mel_DTC.pkl')
            self.__rfc_clf = joblib.load(path + 'Mel_RFC.pkl')

    def __trainAll(self, X, Y):
        return ((self.__svm_clf).fit(X, Y), (self.__nusvm_clf).fit(X, Y), (self.__linsvm_clf).fit(X, Y), (self.__mlpc_clf).fit(X, Y), (self.__dtc_clf).fit(X, Y), (self.__rfc_clf).fit(X, Y))

    def __saveModelsToFile(self, path):
        joblib.dump(self.__svm_clf, (path + 'Mel_SVM.pkl'))
        joblib.dump(self.__nusvm_clf, (path + 'Mel_NuSVM.pkl'))
        joblib.dump(self.__linsvm_clf, (path + 'Mel_LinSVM.pkl'))
        joblib.dump(self.__mlpc_clf, (path + 'Mel_MLPC.pkl'))
        joblib.dump(self.__dtc_clf, (path + 'Mel_DTC.pkl'))
        joblib.dump(self.__rfc_clf, (path + 'Mel_RFC.pkl'))

    def predicto(self, extfeatarr, supresults):
        svm_res = (self.__svm_clf).predict(list(extfeatarr))
        nusvm_res = (self.__nusvm_clf).predict(list(extfeatarr))
        linsvm_res = (self.__linsvm_clf).predict(list(extfeatarr))
        mlpc_res = (self.__mlpc_clf).predict(list(extfeatarr))
        dtc_res = (self.__dtc_clf).predict(list(extfeatarr))
        rfc_res = (self.__rfc_clf).predict(list(extfeatarr))
        dct = {
                    'SVM' : { 'Prediction Results' : svm_res, 'Accuracy' : accuracy_score(list(supresults), svm_res)},
                    'NuSVM': {'Prediction Results': nusvm_res, 'Accuracy': accuracy_score(list(supresults), nusvm_res)},
                    'LinSVM': {'Prediction Results': linsvm_res, 'Accuracy': accuracy_score(list(supresults), linsvm_res)},
                    'MLPC': {'Prediction Results': mlpc_res, 'Accuracy': accuracy_score(list(supresults), mlpc_res)},
                    'DTC': {'Prediction Results': dtc_res, 'Accuracy': accuracy_score(list(supresults), dtc_res)},
                    'RFC': {'Prediction Results': rfc_res, 'Accuracy': accuracy_score(list(supresults), rfc_res)}
              }
        print(dct)
        return dct
