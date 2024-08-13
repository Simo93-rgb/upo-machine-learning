# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:30:01 2019

@author: Luigi Portinale
"""

import numpy as np
import numpy.linalg as la
import pandas as pd

#np array
a=np.array([[1,2],[3,4]])
print(a)
print(a.shape)
(n_row,n_col)=a.shape
print(n_row)
print(n_col)
print(a[1])
#
##operations
x = np.array([1,5,2])
y = np.array([7,4,1])
print(x+y)
#elementwise operations
print(x*y)
#
#dot/inner product
print('Dot prod: ', np.dot(x,y))
#
m1 = np.matrix( ((2,3), (3, 5)) )
m2 = np.matrix( ((1,2), (5, -1)) )
##if matrix we can use *
m3=m1*m2
print('Dot prod: ', m3)
#
#if array use np.dot
m1 = np.array( ((2,3), (3, 5)) )
m2 = np.array( ((1,2), (5, -1)) )
m3=np.dot(m1,m2)
print('Dot prod: ', m3)
m4=m1*m2
print('Elementwise: ', m4)
#
# inverting
im3=la.inv(m3)
print('Inverse: ', im3)

#determinant
det=la.det(m3)
print('Determinant: ',det)

#norma
print('Norm :',la.norm(x))
#
#
#
# dataframe
d={'col1':[1,2,3], 'col2':[4,5,6]}
df=pd.DataFrame(data=d)
print(df.columns[0])
#
#from np array
df1=pd.DataFrame(a,columns=['c1', 'c2'])
#
print(df.at[1,'col2'])
#
print(df.values)
print(df.loc[0])
#
print(df.loc[0:1,['col2']])
#
df2=pd.DataFrame(a,columns=['c1','c2'],index=['a','b'])