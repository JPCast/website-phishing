from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from numpy import sort
from sklearn.feature_selection import SelectFromModel


data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:9]
y = data[:,9]

seed = 7
testAndCrossValidation_size = 0.45
X_train, X_crossAndTest, y_train, y_crossAndTest = train_test_split(X, y, test_size=testAndCrossValidation_size, shuffle=False)
test_size=0.428571429
X_CrossValidation, X_test, y_CrossValidation, y_test = train_test_split(X_crossAndTest, y_crossAndTest, test_size=test_size, shuffle=False)

# fit model no training data
model = XGBClassifier()
#print(model)
model.fit(X_train, y_train)
# plot the importance of feautures
plot_importance(model)
pyplot.show()


#make predictions
#y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
#verify predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

########################################################train data by test each subset of features by importance.

max_depths=range(3,10)   #=4

learning_rates=[0.5]  #=0.5
learning_rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] #=0.5

n_estimators=[80] #=80   n conlusion
n_estimators=range(50,200,10) #=50   n conlusion

#boosters=['gbtree','gblinear','dart']
min_child_weights=[1]  #=1
min_child_weights=range(1,4)  #=3

saveModel=[]
accuracys=[]
for max_depth in max_depths:
	for learning_rate in learning_rates:
		for n_estimator in n_estimators:
			for min_child_weight in min_child_weights:
				model =  XGBClassifier(min_child_weight=min_child_weight,max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimator)
				model.fit(X_train, y_train)
				y_pred = model.predict(X_CrossValidation)
				predictions = [round(value) for value in y_pred]
				accuracy = accuracy_score(y_CrossValidation, predictions)
				accuracys.append([accuracy])
				saveModel.append([min_child_weight,max_depth,learning_rate,n_estimator])
				#print(model)
				#print("Accuracy: %.2f%%" % (accuracy*100.0))

lastMax=accuracys[accuracys.index(max(accuracys))]
count=0
while count<10:
	if len(accuracys)>1 and len(saveModel)>1:
		ind=accuracys.index(max(accuracys))
		if lastMax!=accuracys[ind]:
			count=count+1
			lastMax==accuracys[ind]
		print(accuracys.pop(ind))
		print(saveModel.pop(ind))
