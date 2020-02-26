import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# block of code to import the classes from the parent directory 
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from DecisionTreeClassifier import DecisionTreeClassifier as dtc 
from randomForest import RandomForest


iris = load_iris()
pX = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
py = pd.DataFrame(iris.target, columns =["Species"])


X= pX.to_numpy()
Y = py.to_numpy().flatten()

indices = np.arange(150)
np.random.shuffle(indices)
train_dx , test_idx = indices[:100] , indices[100:]
Xtrain,Ytrain = X[train_dx] , Y[train_dx]
Xtest,Ytest = X[test_idx] , Y[test_idx]

# test of decision tree 
# model = dtc(6,10)
# model.build_tree(Xtrain,Ytrain)
# print('-------------------------------------------------------------------------------------------------------')

# predicted_calsses = model.predict(Xtest)

# score= 0
# for i in range(len(predicted_calsses)):
#     score += (predicted_calsses[i]==Ytest[i])
    
# print("the algorithm has an accuracy of ",score/len(predicted_calsses))


# test of random forest 
m = RandomForest(3,2,5)
m.build_forest(Xtrain,Ytrain)

predicted_calsses = m.predict(Xtest)

score= 0
for i in range(len(predicted_calsses)):
    score += (predicted_calsses[i]==Ytest[i])
    
print("the algorithm has an accuracy of ",score/len(predicted_calsses))











