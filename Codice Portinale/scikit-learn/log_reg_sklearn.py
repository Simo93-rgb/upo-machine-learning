# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:07:39 2018

@author: Luigi Portinale
"""

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics #for confusion matrix
from sklearn.model_selection import train_test_split #for train/test split
from sklearn.model_selection import cross_val_score #for cross validation
from sklearn.metrics import classification_report # for a complete classification report

# LETTURA DATASET IN FORMATO CSV
#dataset=pd.read_csv('breast_cancer/breast-cancer-wisconsin_no_missing_val.csv')
#dataset=pd.read_csv('banknote_authentication.csv')
dataset=pd.read_csv('iris/iris.csv')

#ESTRAZIONE CLASSE y
target=dataset['Class']
y=np.asarray(target)
dataset=dataset.drop('Class',axis=1)

#CREZIONE VETTORE FEATURES

x = dataset.values #TRAINING INSTANCES


log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(x,y)

r1=log_reg.predict(x[39:51]) #predice classe per esempi dal 39 al 51
r2=log_reg.predict(x[30].reshape(1,-1)) # predice classe per es. 39 
                                     # reshape perche' predict vuole array 2D
                                     
r_all=log_reg.predict(x) # all dataset
                                     
# Use score method to get accuracy of model
score = log_reg.score(x, y)
print(score)

##confusion matrix
cm = metrics.confusion_matrix(y, r_all)
print(cm)

#### splitting train/test

train, test, train_lbl, test_lbl = train_test_split(
 x, y, test_size=1/7.0, random_state=0)

log_reg.fit(train,train_lbl)

pred=log_reg.predict(test)
cm = metrics.confusion_matrix(test_lbl, pred)
print(cm)

score = log_reg.score(test, test_lbl)
print(score)

# cross validation on all dataset
scores = cross_val_score(log_reg, x, y, cv=10)
print(scores)

#quick way to print a 95% CI
print("Accuracy (95%%): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#complete classification report
print(classification_report(test_lbl, pred)) # support is the number of instances of each class
