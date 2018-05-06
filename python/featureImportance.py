from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('PhishingData.txt', delimiter=",")
print(len(dataset[0]))
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,9]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()
