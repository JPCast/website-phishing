from sklearn import svm
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:8]
y = data[:,9]

seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#one vs rest
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf)
y_pred = clf.predict(X_test)

predictions = [round(value) for value in y_pred]
#verify predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
