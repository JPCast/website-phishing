from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import sys


print("------------------------------------------------")
print("to test this algoithm using polynomial feautures use as argument -> -p as flag with the number of feautures that you want use -> -p 2")


usePolynomialFeatures=False
if len(sys.argv)>2:
	if str(sys.argv[1])=='-p':
		usePolynomialFeatures=True
		numberOfPolynomialFeautures=int(str(sys.argv[2]))


data = loadtxt('PhishingData.txt', delimiter=",")

# split data into X and y
X = data[:,0:9]
y = data[:,9]

if usePolynomialFeatures:
	poly = PolynomialFeatures(numberOfPolynomialFeautures)
	X=poly.fit_transform(X)

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
numbersOfFeatures=[]
accuracys=[]
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_CrossValidation)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_CrossValidation, predictions)
	numbersOfFeatures.append(select_X_train.shape[1])
	accuracys.append(accuracy*100)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
plt.plot(numbersOfFeatures,accuracys)

if len(sys.argv)>2:
	if str(sys.argv[1])=='-p':
		plt.title('Gradient Boosted Trees - Polynomial Degree '+str(sys.argv[2])) # subplot 211 title
else:
	plt.title('Gradient Boosted Trees - Default Arguments') # subplot 211 title
t = plt.xlabel('Number of Feautures', fontsize=12)
t = plt.ylabel('Accuracy', fontsize=12)

plt.show()
