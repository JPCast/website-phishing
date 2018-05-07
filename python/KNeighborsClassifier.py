from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:9]
y = data[:,9]


seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

for i in range(1,10):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)


    y_pred = neigh.predict(X_test)

    predictions = [round(value) for value in y_pred]
    #verify predictions
    accuracy = accuracy_score(y_test, predictions)
    print("n_neighbors=%s"%(i))
    print("Accuracy: %.2f%% \n" % (accuracy * 100.0))
