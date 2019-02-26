#------- Loading libraries
import sys
import scipy
import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#------- Loading dataset
url = "C:/Desktop/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#------- Summarizing data
#Shape
print(dataset.shape)
#Head
print(dataset.head(20)) # First 20 rows of data
#Descriptions
print(dataset.describe())
#Class distribution
print(dataset.groupby('class').size())

#------- Visualization
#Univariate box & whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey= False)
plt.show()
#Univariate histogram
dataset.hist()
plt.show()
#Multivariate attribute scatter plot
scatter_matrix(dataset)
plt.show()

#------- Algorithm Evaluation
#Create validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Set up test harness to use 10 fold cross validation
scoring = 'accuracy'
#Build 5 diff models to for species prediction
models=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results=[]
names=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#Choose best model
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#------- Making Predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
