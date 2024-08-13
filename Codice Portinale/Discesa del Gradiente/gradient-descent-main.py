# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:11:51 2017

@author: L. Portinale
"""

import numpy as np
import pandas as pd
#from math import log, exp
import math
import sys


#################################################
# 
# FUNCTIONS FOR LOGISTIC REGRESSION / gradient descent
#
#################################################

def compute_hypothesis(theta,x):
    '''
    calcolo del valore di h(x)
    '''
    
    exponent=np.inner(theta,x)
    return 1/(1+math.exp(-exponent))

    

####

def compute_cost_f(theta,x,y,m):
    '''
    compute and return the cost function J(theta) over m instances
    '''
    cost=0
    
    for i in range(0,m):
        h=compute_hypothesis(theta,x[i])
        if h==0:
            h=1.0E-15
        elif h==1:
            h=0.999999999999999
        l1=math.log(h)
        l2=math.log(1-h)
        cost=cost+y[i]*l1+(1-y[i])*l2
    
    cost=cost*(-1/m)
    return cost

###
    
def compute_derivative(j,theta,x,y,m):
    '''
    compute the derivative of the cost function
    wrt theta[j]
    '''
    der=0
    
    for i in range(0,m):
        h=compute_hypothesis(theta,x[i])
        #print('h= ',h)
        der=der+(h-y[i])*x[i,j]
        
    return der

###

def gradient_descent(alpha,tolerance,theta,x,y,m):
    
    pcost=compute_cost_f(theta,x,y,m)
    
    print('Cost= ',pcost)
    
    temp=np.copy(theta) # no assegnamento altrimenti copia la reference
    
    nstep=1
    
    while True:
        
        for j in range(0,len(theta)):        
            temp[j]=theta[j]-alpha*compute_derivative(j,theta,x,y,m)
            
        theta=np.copy(temp)
        
        #print('Theta= ',theta)
        
        cost=compute_cost_f(theta,x,y,m)
        
        print('Cost= ', cost)
        
        precision=pcost-cost
        if precision<tolerance:
        #if nstep==8000:
            break
        else:           
            #theta=np.copy(temp)
            pcost=cost
            nstep=nstep+1
    
    print('N. Steps= ',nstep)
    
    return (theta,cost)

###
    
def predict(theta,x,flag):
    '''
    predict the class of vector x using parameters theta
    flag=0 print off
    flag=1 print on
    return tuple (class, probability)
    '''
    
    h=compute_hypothesis(theta,x)
    
    if h>0.5:
        if flag==1:
            print('Predicted class: 1, with probability ',h)
        return (1,h)
    else:
        if flag==1:
            print('Predicted class: 0, with probability ',1-h)
        return (0,1-h)



####

def evaluate_test(file,theta):
    '''
    evaluate the model theta with the test set in file
    '''
    testset=pd.read_csv(file)
    #EVENTUALE ELIMINAZIONE DI CAMPI INUTILI (ES: ID, CODE, ECC..)
    #dataset=dataset.drop('SampleCode', axis=1) #axis=0 row; axis=1 column

    target=testset['Class']
    y=np.asarray(target)
    testset=testset.drop('Class',axis=1)

    #CREZIONE VETTORE FEATURES
    values = testset.values #TEST INSTANCES
    m=len(values) # NUMBER OF TEST INSTANCES

    vl=values.tolist() #PASSO A STRUTTURA LISTA PER AGGIUNGERE 1 IN TESTA AD OGNI ISTANZA

    for i in range(0,m):
        vl[i].insert(0,1)

    # CREAZIONE VETTORE DELLE ISTANZE x
    x=np.asarray(vl)
    ### INITIALIZE STATISTICS
    tp=0 #TRUE PPOSITIVE
    tn=0 #TRUE NEGATIVE
    fn=0 #FALSE NEGATIVE
    fp=0 #FALSE POSITIVE
    
    # INITIALIZE PROBABILITY OF PREDICTION pr[i]=prob classe istanza i
    pr=np.zeros(m)
    # INITIALIZE VECTOR OF PREDICTIONS cl[i]=predicted classe istanza i
    cl=np.zeros(m)
    
    # LOOP FOR EVALUATION THROUGH TEST SET
    for i in range(0,m):
        (cl[i],pr[i])=predict(theta,x[i],0)
                
        if cl[i]==0:
            if cl[i]==y[i]:
                tn=tn+1
            else:
                fn=fn+1
        elif cl[i]==1:
            if cl[i]==y[i]:
                tp=tp+1
            else:
                fp=fp+1
    
    # PRINT PERFORMANCE MEASURES
    accuracy=(tp+tn)/m
    print('Accuracy: ', accuracy)
    
    if(tp+fn==0):
        print('No TP and FN')
    else:
        recall=tp/(tp+fn)
        print('Recall: ', recall)
    
    if(tp+fp==0):
        print('No TP and FP')
    else:
        precision=tp/(tp+fp)
        print('Precision: ', precision)
    
    f1score=(2*tp)/(2*tp+fp+fn)
    print('F1 score: ', f1score)
    
    
    #PRINT PREDICTIONS
    old_out=sys.stdout
    sys.stdout = open('output.txt','w')
    for i in range(0,m):
        print('Instance ', i, 'Class: ',cl[i], 'Prob: ', pr[i])
        
    sys.stdout.close()
    sys.stdout=old_out