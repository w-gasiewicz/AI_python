import pandas
import numpy as np
import sklearn.feature_selection as feat_select
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

# read data
dataframe = pandas.read_excel('ANEMIA.XLS')
array = dataframe.values
X = array[:, 2:33]  # features columns
Y = array[:, 0]  # class column
it = 0
class_nr = 1.0
for i in Y:  # fill empty cells with the class number
    if np.isnan(i):
        Y[it] = class_nr
    else:
        class_nr = i
    it += 1
# feature scoring
test = feat_select.SelectKBest(score_func=feat_select.chi2, k='all')
fit = test.fit(X, Y.astype('int'))
# print scores
#scores = []
#for i in range(31):
#    scores.append([i, fit.scores_[i]])
#scores = sorted(scores, key=lambda item: item[1], reverse=True)
#print(scores)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, random_state=27)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#momentum
#mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=900, alpha=0.0001,momentum=0.9, nesterovs_momentum=True,
#                    solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

           #no momentum          
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=900, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
                     
mlp.fit(X_train,y_train.astype('int'))
prediction = mlp.predict(X_test)

print(prediction)
print(confusion_matrix(y_test.astype('int'),prediction))
print(mlp.score(X_test,y_test.astype('int'),prediction))
print(classification_report(y_test.astype('int'), prediction))