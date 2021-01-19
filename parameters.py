import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score 
from sklearn.model_selection import cross_val_score


#Importing the Dataset
#-----------------------------------------------------------------------------------------------
dataset = pd.read_csv('Datasets/Data.csv')
#X = dataset.iloc[:, :-1].values   # X --> values of the independent variable columns as a matrix 
#y = dataset.iloc[:, -1].values    # y --> values of the dependent variable column as a vector
X = dataset.iloc[:50,:-1].values   
y = dataset.iloc[:50,-1].values



self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
self.regressor = DecisionTreeRegressor(random_state = 0) 
self.regressor.fit(self.X_train, self.y_train) 
self.y_pred = self.regressor.predict(self.X_test)
self.r2_score = r2_score(self.y_test,self.y_pred)
