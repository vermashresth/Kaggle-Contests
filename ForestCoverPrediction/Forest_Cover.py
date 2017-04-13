# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:39:22 2017

@author: Shresth
"""

import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
forest=pandas.read_csv("train.csv")
#print forest.describe()
alg=LogisticRegression(random_state=1)
predictors=list(forest.axes[1])
soils=list(forest.columns[15:55])
df=forest[forest["Cover_Type"]==2][soils].mean()
temp=[]
mainSoils=set(temp)
for i in range(1,8):
    newy=forest[forest["Cover_Type"]==i][soils].mean()
    newy.sort()
    newx=newy[-7:]
    newh=list(newx.axes[0])
    temp.extend(newh)
mainSoils=set(temp)
print mainSoils
pred=predictors[0:14]
pred.extend(predictors[55:])



selectors=SelectKBest(f_classif,k=5)
selectors.fit(forest[predictors],forest["Cover_Type"])
scores=-np.log10(selectors.pvalues_)
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')
plt.show()

desc=pandas.DataFrame({
        "Features":forest.columns,
        "Values":scores})
print desc[desc["Values"]>150]["Features"].

print scores

#scores=cross_validation.cross_val_score(alg,forest[predictors],forest["Cover_Type"],cv=3)
#print scores.mean()