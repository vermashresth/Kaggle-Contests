# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:56:13 2017

@author: Shresth
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation 

train=pd.read_csv("train.csv")

#correlation
size=10
numcols=train.columns[:10]
numdata=train[numcols]
data_corr=numdata.corr()
threshold=.5
corr_list = []
cols=numdata.columns 
#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
for v,i,j in s_corr_list:
    sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()
    
    