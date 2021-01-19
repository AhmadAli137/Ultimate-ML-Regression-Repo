#Importing the Libraries
#-------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 
import numpy as np

class SVR_Reg:
	def __init__(self, X, y):

		#Splitting the dataset into the Training set and Test set
		#-----------------------------------------------------------------------------------------------
		y = y.reshape(len(y),1)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
		# X_train --> matrix of dependent values for training set
		# X_test --> matrix of dependent values for test set
		# y_train --> vector of independent values for training set
		# y-test --> vector of independent values for test set
		# test_size --> percentage of original data removed to form test set
		# random_state --> the seed which randomly partitions the dataset into training and test sets
		# Note: The ideal split is 80:20 for training set:test set. However, 
		#       this only applies when the dataset is large. 

		#Feature Scaling
		#-----------------------------------------------------------------------------------------------
		self.sc_X = StandardScaler()
		self.sc_y = StandardScaler()
		self.X_train = self.sc_X.fit_transform(self.X_train)
		self.y_train = self.sc_y.fit_transform(self.y_train)

		#Training the Support Vector Regression model on the Training set
		#-----------------------------------------------------------------------------------------------
		self.regressor = SVR(kernel = 'rbf') # regressor refers to the model we will use to make a prediction
		# kernel is the function used to map a lower dimensional data into a higher dimensional data.
		# rbf = radial basis function
		self.regressor.fit(self.X_train, self.y_train.ravel()) # trains the regressor on the training set data

		#Predicting the Test set results
		#-----------------------------------------------------------------------------------------------
		self.y_pred = self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(self.X_test)))
		#returns predictions of the target values in the test set based on the 
		#               independent variables in the test set

		#Evaluating the Model Perfomance
		#-----------------------------------------------------------------------------------------------
		self.r2_score = r2_score(self.y_test,self.y_pred)
		# returns the R-squared value of the newly formed regression model
		# The R-squared value is a measure of how much better the regression model is able to predict 
		#     the target values vs using a statistical average 
		# The closer the R-squared value to 1, the more accurate the regression model is