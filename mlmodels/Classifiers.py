from sklearn.externals import joblib
from sklearn.svm import SVC, SVR
from sklearn.svm import NuSVC, NuSVR
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score

class Classifiers(object):

    def __init__(self, featureset=None, target=None, mode='predict', path=''):
        if (mode == 'train'):
            self.__svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            self.__svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            self.__nusvm = NuSVC(cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            self.__nusvr = NuSVR(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, nu=0.5, shrinking=True, tol=0.001, verbose=False)
            self.__linsvm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
            self.__linsvr = LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True, intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000, random_state=None, tol=0.0001, verbose=0)
            self.__mlpc = MLPC(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(100, 25), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
            self.__mlpr = MLPR(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(100, 25), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
            self.__dtc = DTC(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
            self.__dtr = DTR(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
            self.__rfc = RFC(bootstrap=True, class_weight=None, criterion='gini', max_depth=100, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
            self.__rfr = RFR(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
            (self.__svm, self.__svr, self.__nusvm, self.__nusvr, self.__linsvm, self.__linsvr, self.__mlpc, self.__mlpr, self.__dtc, self.__dtr, self.__rfc, self.__rfr) = self.__trainAll(X=list(featureset), Y=list(target))
            self.__saveModelsToFile(path)
        else:
            self.__svm = joblib.load(path + 'Mel_SVM.pkl')
            self.__svr = joblib.load(path + 'Mel_SVR.pkl')
            self.__nusvm = joblib.load(path + 'Mel_NuSVM.pkl')
            self.__nusvr = joblib.load(path + 'Mel_NuSVR.pkl')
            self.__linsvm = joblib.load(path + 'Mel_LinSVM.pkl')
            self.__linsvr = joblib.load(path + 'Mel_LinSVR.pkl')
            self.__mlpc = joblib.load(path + 'Mel_MLPC.pkl')
            self.__mlpr = joblib.load(path + 'Mel_MLPR.pkl')
            self.__dtc = joblib.load(path + 'Mel_DTC.pkl')
            self.__dtr = joblib.load(path + 'Mel_DTR.pkl')
            self.__rfc = joblib.load(path + 'Mel_RFC.pkl')
            self.__rfr = joblib.load(path + 'Mel_RFR.pkl')

    def __trainAll(self, X, Y):
        return ((self.__svm).fit(X, Y), (self.__svr).fit(X, Y), (self.__nusvm).fit(X, Y), (self.__nusvr).fit(X, Y), (self.__linsvm).fit(X, Y), (self.__linsvr).fit(X, Y), (self.__mlpc).fit(X, Y), (self.__mlpr).fit(X, Y), (self.__dtc).fit(X, Y), (self.__dtr).fit(X, Y), (self.__rfc).fit(X, Y), (self.__rfr).fit(X, Y))

    def __saveModelsToFile(self, path):
        joblib.dump(self.__svm, (path + 'Mel_SVM.pkl'))
        joblib.dump(self.__svr, (path + 'Mel_SVR.pkl'))
        joblib.dump(self.__nusvm, (path + 'Mel_NuSVM.pkl'))
        joblib.dump(self.__nusvr, (path + 'Mel_NuSVR.pkl'))
        joblib.dump(self.__linsvm, (path + 'Mel_LinSVM.pkl'))
        joblib.dump(self.__linsvr, (path + 'Mel_LinSVR.pkl'))
        joblib.dump(self.__mlpc, (path + 'Mel_MLPC.pkl'))
        joblib.dump(self.__mlpr, (path + 'Mel_MLPR.pkl'))
        joblib.dump(self.__dtc, (path + 'Mel_DTC.pkl'))
        joblib.dump(self.__dtr, (path + 'Mel_DTR.pkl'))
        joblib.dump(self.__rfc, (path + 'Mel_RFC.pkl'))
        joblib.dump(self.__rfr, (path + 'Mel_RFR.pkl'))

    def predicto(self, extfeatarr, supresults):
        svm_res = (self.__svm).predict(list(extfeatarr))
        svr_res = (self.__svr).predict(list(extfeatarr))
        nusvm_res = (self.__nusvm).predict(list(extfeatarr))
        nusvr_res = (self.__nusvr).predict(list(extfeatarr))
        linsvm_res = (self.__linsvm).predict(list(extfeatarr))
        linsvr_res = (self.__linsvr).predict(list(extfeatarr))
        mlpc_res = (self.__mlpc).predict(list(extfeatarr))
        mlpr_res = (self.__mlpr).predict(list(extfeatarr))
        dtc_res = (self.__dtc).predict(list(extfeatarr))
        dtr_res = (self.__dtr).predict(list(extfeatarr))
        rfc_res = (self.__rfc).predict(list(extfeatarr))
        rfr_res = (self.__rfr).predict(list(extfeatarr))
        return ({
                    'SVM' : { 'Prediction Results' : svm_res, 'Accuracy' : accuracy_score(list(supresults), svm_res)},
                    'SVR': {'Prediction Results': svr_res, 'Accuracy': explained_variance_score(list(supresults), svr_res)},
                    'NuSVM': {'Prediction Results': nusvm_res, 'Accuracy': accuracy_score(list(supresults), nusvm_res)},
                    'NuSVR': {'Prediction Results': nusvr_res, 'Accuracy': explained_variance_score(list(supresults), nusvr_res)},
                    'LinSVM': {'Prediction Results': linsvm_res, 'Accuracy': accuracy_score(list(supresults), linsvm_res)},
                    'LinSVR': {'Prediction Results': linsvr_res, 'Accuracy': explained_variance_score(list(supresults), linsvr_res)},
                    'MLPC': {'Prediction Results': mlpc_res, 'Accuracy': accuracy_score(list(supresults), mlpc_res)},
                    'MLPR': {'Prediction Results': mlpr_res, 'Accuracy': explained_variance_score(list(supresults), mlpr_res)},
                    'DTC': {'Prediction Results': dtc_res, 'Accuracy': accuracy_score(list(supresults), dtc_res)},
                    'DTR': {'Prediction Results': dtr_res, 'Accuracy': explained_variance_score(list(supresults), dtr_res)},
                    'RFC': {'Prediction Results': rfc_res, 'Accuracy': accuracy_score(list(supresults), rfc_res)},
                    'RFR': {'Prediction Results': rfr_res, 'Accuracy': explained_variance_score(list(supresults), rfr_res)}
                })