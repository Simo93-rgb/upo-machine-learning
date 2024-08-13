# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:11:51 2017

@author: L. Portinale
"""

import numpy as np
import pandas as pd
import logistic_regression as lr

# LETTURA DATASET IN FORMATO CSV
#dataset=pd.read_csv('breast-cancer-wisconsin_no_missing_val.csv')
#dataset=pd.read_csv('banknote_authentication.csv')
dataset=pd.read_csv('iris/virginica_vs_other.csv')

#print(dataset)

#EVENTUALE ELIMINAZIONE DI CAMPI INUTILI (ES: ID, CODE, ECC..)
#dataset=dataset.drop('SampleCode', axis=1) #axis=0 row; axis=1 column

#print(dataset)

#ESTRAZIONE NUMERO FEATURES/ATTRIBUTI DA UTILIZZARE
nf=len(dataset.columns)


#######
# CREAZIONE VETTORE PARAMETRI INIZIALI
rng=np.random
theta=rng.randn(nf) #from standard normal distribution
#theta=np.zeros(nf) #theta[0] ....theta[nf-1]
#theta[0]=-7.32
#theta[1]=7.86
#theta[2]=4.19
#theta[3]=5.30
#theta[4]=0.605
#ESTRAZIONE CLASSE y
target=dataset['Class']
y=np.asarray(target)
dataset=dataset.drop('Class',axis=1)

#CREZIONE VETTORE FEATURES

values = dataset.values #TRAINING INSTANCES
m=len(values) # NUMBER OF TRAINING INSTANCES

vl=values.tolist() #PASSO A STRUTTURA LISTA PER AGGIUNGERE 1 IN TESTA AD OGNI ISTANZA

for i in range(0,m):
    vl[i].insert(0,1)

# CREAZIONE VETTORE DELLE ISTANZE x
x=np.asarray(vl)

# x[i,j] FEATURE j DEL CASO/ISTANZA i

# # PER GESTIRE POSIZIONI FEATURES A LORO NOMI
# index_names = dataset.index
# col_names = dataset.columns
# col_names[index_names[i]]=nome feature in posizione i

#cancer   
#(theta,cost)=lr.gradient_descent(0.000000000000001,0.000000000005,theta,x,y,m)
# banknote
#(theta,cost)=lr.gradient_descent(0.00001,0.00005,theta,x,y,m)
# iris virginica
(theta,cost)=lr.gradient_descent(0.0001,0.000005,theta,x,y,m)

#evaluation on the training set
print('Evaluating on the training set >>>> \n')
#lr.evaluate_test('banknote_authentication.csv',theta)
#lr.evaluate_test('breast_cancer/breast-cancer-wisconsin_no_missing_val.csv',theta)
lr.evaluate_test('iris/virginica_vs_other.csv',theta)