import sys
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_importance
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import loadtxt
from matplotlib import pyplot
from numpy import sort

print("---------------------------------------------------------")
print("to test svm one vs rest with the best feautures calculated by xgboost use as argument -> -m")


useMostImportantFeature=False
if len(sys.argv)>1:
     if str(sys.argv[1])=='-m':
         useMostImportantFeature=True



data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:9]
y = data[:,9]

seed = 7
testAndCrossValidation_size = 0.45
X_train, X_crossAndTest, y_train, y_crossAndTest = train_test_split(X, y, test_size=testAndCrossValidation_size, shuffle=False)
test_size=0.428571429
X_CrossValidation, X_test, y_CrossValidation, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

#one vs rest
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf)
y_pred = clf.predict(X_CrossValidation)

predictions = [round(value) for value in y_pred]
#verify predictions
accuracy = accuracy_score(y_CrossValidation, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


########################################################################

if useMostImportantFeature:
    # fit model no training data -> only used to calculat the importance of features
    model = XGBClassifier()
    #print(model)
    model.fit(X_train, y_train)
    # plot the importance of feautures
    plot_importance(model)
    pyplot.show()
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
    	# select features using threshold
    	selection = SelectFromModel(model, threshold=thresh, prefit=True)
    	select_X_train = selection.transform(X_train)
    	# train model
    	selection_model = svm.SVC()
        selection_model.fit(select_X_train, y_train)
    	# eval model
    	select_X_test = selection.transform(X_CrossValidation)
    	y_pred = selection_model.predict(select_X_test)
    	predictions = [round(value) for value in y_pred]
    	accuracy = accuracy_score(y_CrossValidation, predictions)
    	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
