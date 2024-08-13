# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:07:39 2018

@author: Gigi
"""

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split #for train/test split
#from sklearn.preprocessing import Imputer #for missing values


# LETTURA DATASET IN FORMATO CSV

dataset=pd.read_csv('auto-mpg/auto-mpg_nomiss.csv')

#ESTRAZIONE CLASSE y
target=dataset['MPG']
y=np.asarray(target)
dataset=dataset.drop('MPG',axis=1)

#CREZIONE VETTORE FEATURES

x = dataset.values #TRAINING INSTANCES

#imputing missing values
#imputer = Imputer(missing_values='?')
#x = imputer.fit_transform(x)


lin_reg=LinearRegression()
lin_reg.fit(x,y)

r1=lin_reg.predict(x[39:51]) #predice classe per esempi dal 39 al 51
r2=lin_reg.predict(x[30].reshape(1,-1)) # predice classe per es. 39 
                                     # reshape perche' predict vuole array 2D
                                     
r_all=lin_reg.predict(x) # all dataset
                                     
# Use score method to get accuracy of model
score = lin_reg.score(x, y)
print(score)


#### splitting train/test

train, test, train_lbl, test_lbl = train_test_split(
 x, y, test_size=1/7.0, random_state=0)

lin_reg.fit(train,train_lbl)

pred=lin_reg.predict(test)

# The coefficients but the intercept
print('Coefficients: \n', lin_reg.coef_)
# The intercept Theta[0]
print('Intercept: \n', lin_reg.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(test_lbl, pred))
# R2 score: 1- % of explained variance of the model
print('R2 score: %.3f' % r2_score(test_lbl, pred))
# explained variance score: same as before but by taking mean_error into account 
print('Var score: %.3f' % explained_variance_score(test_lbl, pred))

## Plot outputs
#plt.scatter(test, test_lbl,  color='black')
#plt.plot(test, pred, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
#
# method score produce R2
score = lin_reg.score(test, test_lbl)
print(score)