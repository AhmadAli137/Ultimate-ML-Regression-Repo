#sources
#-----------------------------------------------------------------------------------------------
#"Machine Learning A-Z™: Hands-On Python & R In Data Science" course content taught on Udemy by Kirill Eremenko & Hadelin de Ponteves
#https://howtothink.readthedocs.io/en/latest/PvL_H.html
#https://stackoverflow.com/questions/30336138/how-to-plot-a-multivariate-function-in-python

#Importing the Libraries
#-----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from Multiple_Linear_Regression_Algorithm import *
from Polynomial_Regression_Algorithm import *
from Support_Vector_Regression_Algorithm import *
from Decision_Tree_Regression_Algorithm import *
from Random_Forest_Regression_Algorithm import *

#Importing the Dataset
#-----------------------------------------------------------------------------------------------
dataset = pd.read_csv('Datasets/Data.csv')
#X = dataset.iloc[:, :-1].values   # X --> values of the independent variable columns as a matrix 
#y = dataset.iloc[:, -1].values    # y --> values of the dependent variable column as a vector
X = dataset.iloc[:50,:-1].values   
y = dataset.iloc[:50,-1].values

#Creating and Applying the Regressor on dataset
#-----------------------------------------------------------------------------------------------
MLR_reg = MLR(X,y)
PR_reg = PR(X,y)
SVR_reg = SVR_Reg(X,y)
DTR_reg = DTR(X,y)
RFR_reg = RFR(X,y)

def Get_Model():
	#Modelling the predictions of the Regressor vs actual values
	#-----------------------------------------------------------------------------------------------
	plt.plot(MLR_reg.y_pred, color = 'blue', label='Multiple Linear (r^2 score = '+str('%.3f'%MLR_reg.r2_score)+")")
	plt.plot(PR_reg.y_pred, color = 'green', label='Polynomial (r^2 score = '+str('%.3f'%PR_reg.r2_score)+")")
	plt.plot(SVR_reg.y_pred, color = 'purple', label='Support Vector (r^2 score = '+str('%.3f'%SVR_reg.r2_score)+")")
	plt.plot(DTR_reg.y_pred, color = 'orange', label='Decision Tree (r^2 score = '+str('%.3f'%DTR_reg.r2_score)+")")
	plt.plot(RFR_reg.y_pred, color = 'cyan', label='Random Forest (r^2 score = '+str('%.3f'%RFR_reg.r2_score)+")")
	plt.plot(MLR_reg.y_test, color = 'red', label='actual values')
	#plt.ylim(425, 480)
	#plt.xlim(0, 2**3)
	plt.ylim(450,500)
	plt.xlim(None,5)
	plt.legend()
	plt.title('Regression Algorithm Predictions')
	plt.figtext(0.5, 0.01, 'Above is only a fraction of actual graph', wrap=True, horizontalalignment='center', fontsize=11)
	plt.savefig("Generated Images\\Prediction.png")

def Get_Algorithm():
	#Finding the Optimal Regression Algorithm
	#-----------------------------------------------------------------------------------------------
	Algorithm_list = ["Multiple Linear","Polynomial","Support Vector","Decision Tree","Random Forest"]
	r2_score_list = [MLR_reg.r2_score, PR_reg.r2_score, SVR_reg.r2_score, DTR_reg.r2_score, RFR_reg.r2_score]
	Opt_Algorithm = ""
	Opt_r2_score = 0
	for i in range(len(Algorithm_list)):
		if r2_score_list[i-1] >= Opt_r2_score:
			Opt_r2_score = r2_score_list[i-1]
			Opt_Algorithm = Algorithm_list[i-1]
	return(Opt_Algorithm)

#val1,val2,val3,val4

def Get_MLR_Prediction(*args):
	rlist = [[]]
	for arg in args:
		rlist[0].append(arg)
	return(MLR_reg.regressor.predict(rlist))

def Get_PR_Prediction(*args):
	rlist = [[]]
	for arg in args:
		rlist[0].append(arg)
	return(PR_reg.regressor.predict(PR_reg.poly.transform(rlist)))

def Get_SVR_Prediction(*args):
	rlist = [[]]
	for arg in args:
		rlist[0].append(arg)
	return(SVR_reg.sc_y.inverse_transform(SVR_reg.regressor.predict(SVR_reg.sc_X.transform(rlist))))

def Get_DTR_Prediction(*args):
	rlist = [[]]
	for arg in args:
		rlist[0].append(arg)
	return(DTR_reg.regressor.predict(rlist))

def Get_RFR_Prediction(*args):
	rlist = [[]]
	for arg in args:
		rlist[0].append(arg)
	return(RFR_reg.regressor.predict(rlist))



Get_Model()
print("The optimal regression algorithm for your dataset is the "+Get_Algorithm()+" Regression algorithm")

print("MLR Prediction: "+str(Get_MLR_Prediction(20,50,1000,97)))
print("PR Prediction [optimal degree = "+str(PR_reg.opt_deg)+"]: "+str(Get_PR_Prediction(20,50,1000,97)))
print("SVR Prediction: "+str(Get_SVR_Prediction(20,50,1000,97)))
print("DTR Prediction: "+str(Get_DTR_Prediction(20,50,1000,97)))
print("RFR Prediction: "+str(Get_RFR_Prediction(20,50,1000,97)))

	


