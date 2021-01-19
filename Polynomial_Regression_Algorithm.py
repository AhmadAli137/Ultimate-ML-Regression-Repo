#Importing the Libraries
#--------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class PR:
	def __init__(self, X, y):

		#Splitting the dataset into the Training set and Test set
		#-----------------------------------------------------------------------------------------------
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
		# X_train --> matrix of dependent values for training set
		# X_test --> matrix of dependent values for test set
		# y_train --> vector of independent values for training set
		# y-test --> vector of independent values for test set
		# test_size --> percentage of original data removed to form test set
		# random_state --> the seed which randomly partitions the dataset into training and test sets
		# Note: The ideal split is 80:20 for training set:test set. However, 
		#       this only applies when the dataset is large. 

		#Using Cross-Validation to determine the best polynomial degree (between 1-10) to use
		#-----------------------------------------------------------------------------------------------
		self.deg_scores=[]
		self.deg_range = list(range(1, 11))
		self.opt_deg =0
		self.opt_r2_score =0
		self.regressor = LinearRegression()# regressor refers to the model we will use to make a prediction

		for deg in self.deg_range:
			self.poly = PolynomialFeatures(degree=deg)
			self.X_poly = self.poly.fit_transform(self.X_train)
			 
			self.score = cross_val_score(self.regressor, self.X_poly, self.y_train, cv=10, scoring='r2').mean()
			self.deg_scores.append(self.score)
			if self.score >= self.opt_r2_score:
				self.opt_r2_score=self.score
				self.opt_deg = deg

		#Plotting polynomial degree vs r2 scores
		#-----------------------------------------------------------------------------------------------
		plt.plot(self.deg_range, self.deg_scores)
		plt.xlabel('Value of deg for Polynomial Regression')
		plt.ylabel('Cross-Validated r^2 score')
		plt.savefig("Generated Images\\Polynomial_Regression_Degree_Cross_Validation.png")

		#Training the Polynomial Regression model on the Training set
		#-----------------------------------------------------------------------------------------------
		self.poly = PolynomialFeatures(degree = self.opt_deg) #used to create the polynomial regressor
		self.X_poly = self.poly.fit_transform(self.X_train) #formatting independent variables for use by regressor
		self.regressor.fit(self.X_poly, self.y_train) # trains the regressor on the training set data

		#Predicting the Test set results
		#-----------------------------------------------------------------------------------------------
		self.y_pred = self.regressor.predict(self.poly.transform(self.X_test)) #returns predictions of the target values in the test set
		#                                   based on the independent variables in the test set

		#Evaluating the Model Perfomance
		#-----------------------------------------------------------------------------------------------
		self.r2_score = r2_score(self.y_test,self.y_pred)
		# returns the R-squared value of the newly formed regression model
		# The R-squared value is a measure of how much better the regression model is able to predict 
		#     the target values vs using a statistical average 
		# The closer the R-squared value to 1, the more accurate the regression model is
