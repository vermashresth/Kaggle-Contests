# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:18:33 2017

@author: Shresth
"""
import pandas
forest=pandas.read_csv("train.csv")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np
from sklearn import cross_validation

alg=RandomForestClassifier(random_state=1,n_estimators=20,min_samples_split=4,min_samples_leaf=3)
alg2=LinearRegression(normalize=True)
alg3=GradientBoostingClassifier(random_state=1,n_estimators=20,max_depth=4)
predictors=list(forest.axes[1])

selector = SelectKBest(f_classif, k=9)

selector.fit(forest[predictors], forest["Cover_Type"])

scores = -np.log10(selector.pvalues_)

submission = pandas.DataFrame({
        "Feature": forest.columns,
        "Value": scores
    })

jo=list(submission[submission["Value"]>60]["Feature"][1:])
print jo
kf=cross_validation.KFold(forest.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(alg3,forest[jo],forest["Cover_Type"],cv=kf)
print scores.mean()
forest_test=pandas.read_csv("test.csv")

alg.fit(forest[jo[:-1]],forest["Cover_Type"])
predictions=alg.predict(forest_test[jo[:-1]])

sub=pandas.DataFrame({ "Cover_Type": predictions,
                      "Id": forest_test["Id"]
                     
                     })
newhh=sub[['Id','Cover_Type']]
print newhh
newhh.to_csv("submissiongradboost.csv", index=False)