#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:37:02 2017

@author: pegasus
"""

import pandas
forest=pandas.read_csv("train.csv")
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np


alg=LogisticRegression(random_state=1)

predictors=list(forest.axes[1])

selector = SelectKBest(f_classif, k=9)

selector.fit(forest[predictors], forest["Cover_Type"])

scores = -np.log10(selector.pvalues_)

"""
print scores

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
"""

submission = pandas.DataFrame({
        "Feature": forest.columns,
        "Value": scores
    })

jo=list(submission[submission["Value"]>60]["Feature"][1:])

"""
scaler=preprocessing.MinMaxScaler()

df_std=[]

df_std[jo[:-1]]=scaler.fit_transform(forest[jo[:-1]])

#df_std.drop(["Cover_Type"],1)
df_std=df_std[:-1]

print df_std"""

soils=list(forest.axes[1][15:55])

newy=forest[forest["Cover_Type"]==1][soils]

#print newy.mean()
temp=[]

hi=set(temp)

for i in range(1,8):
        newy=forest[forest["Cover_Type"]==i][soils].mean()
        newy.sort()
        newx=newy[-18:]
        newx=list(newx.axes[0])
        for j in newx:
            temp.append(j)
            
        hi=set(temp)
#print hi
high=list(hi)

pred=predictors[0:14]

pred.extend(high)

pred.extend(predictors[55:])

pred=pred[1:]

pred.sort()
#print forest["Cover_Type"]
#scoreyyy=cross_validation.cross_val_score(alg, forest[pred], forest["Cover_Type"], cv=3)

scores=cross_validation.cross_val_score(alg, forest[jo], forest["Cover_Type"], cv=3)

print scores.mean()

#print scoreyyy.mean()


forest_test=pandas.read_csv("test.csv")

alg.fit(forest[jo[:-1]],forest["Cover_Type"])
predictions=alg.predict(forest_test[jo[:-1]])

sub=pandas.DataFrame({ "Cover_Type": predictions,
                      "Id": forest_test["Id"]
                     
                     })
newhh=sub[['Id','Cover_Type']]
print newhh
newhh.to_csv("submissionforest.csv", index=False)