import sys
from sklearn.ensemble import BaggingClassifier
import numpy as np
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#ativate to calculate best args by bruteForce
calculateBestArgs=False
if len(sys.argv)>1:
     if str(sys.argv[1])=='-a':
         calculateBestArgs=True


data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:9]
y = data[:,9]


seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
if calculateBestArgs:
    n_estimators=range(1,40)
    max_samples=range(1,20)
    bootstraps=[True]
    bootstrap_features=[True,False]
    oob_scores=[False,True]
    warm_starts=[False]
    accuracys=[]
    clfs=[]
    for n_estimator in n_estimators:
        for max_sample in max_samples:
            for bootstrap in bootstraps:
                for bootstrap_feature in bootstrap_features:
                    for oob_score in oob_scores:
                        for warm_start in warm_starts:
                            clf = BaggingClassifier(n_estimators=n_estimator, max_samples=max_sample,bootstrap=bootstrap, bootstrap_features=bootstrap_feature, oob_score=oob_score, warm_start=warm_start)
                            clf.fit(X_train, y_train)


                            print(clf)
                            y_pred = clf.predict(X_test)

                            predictions = [round(value) for value in y_pred]
                            #verify predictions
                            accuracy = accuracy_score(y_test, predictions)
                            accuracys.append(accuracy)
                            clfs.append(clf)
                            print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(max(accuracys))
    print(max(accuracys))
    indMA = accuracys.index(max(accuracys))
    print(clfs[indMA])
else:
    clf=BaggingClassifier(base_estimator=None, bootstrap=True,
         bootstrap_features=False, max_features=1.0, max_samples=6,
         n_estimators=24, n_jobs=1, oob_score=True, random_state=None,
         verbose=0, warm_start=False)
    clf.fit(X_train, y_train)


    print(clf)
    y_pred = clf.predict(X_test)

    predictions = [round(value) for value in y_pred]
    #verify predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
