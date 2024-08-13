# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:28:42 2019
Esempio PCA su dataset Iris e score plot
@author: Luigi Portinale
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#load iris dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])
# print first rows of data
df.head()

# locate features as x and target/classes as y
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values
y = df.loc[:,['target']].values

# Now standardized features values (mean=0, std=1)
x = StandardScaler().fit_transform(x)
# print frist rows of transformed data
pd.DataFrame(data = x, columns = features).head()

# Perform PCA
pca = PCA(n_components=2) #first 2 pc (if not specified all pc are obtained)
principalComponents = pca.fit_transform(x)

#get the dataframe of PC
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
# print first 5 rows (first 5 examples transformed in the space of PC1 and PC2)
principalDf.head(5)

# produce the final transformed dataframe with also the target/classes
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)

######################
# 2D Visualization 
###################

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 25)
ax.set_ylabel('Principal Component 2', fontsize = 25)
ax.set_title('2 Component PCA', fontsize = 30)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#Explained variance of PC
for i in pca.explained_variance_ratio_:
    print('Var: ',i)



