#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:04:17 2017

@author: pegasus
"""
import pandas
titanic=pandas.read_csv("train.csv")



titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
titanic["Cabin"]=titanic["Cabin"].fillna("C85")
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np

predictions=np.concatenate(predictions,axis=0)

predictions[predictions > 0.5]=1
predictions[predictions < 0.5]=0
           
accuracy = sum(predictions[predictions==titanic["Survived"]])/len(predictions)


titanic_test=pandas.read_csv("test.csv")

titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"]=2
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())

alg=LinearRegression()

alg.fit(titanic[predictors],titanic["Survived"])
predictions1=alg.predict(titanic_test[predictors])
predictions1[predictions1 > 0.5]=int(1)
predictions1[predictions1 < 0.5]=int(0)
submission =pandas.DataFrame({"PassengerId":titanic_test["PassengerId"],"Survived":predictions1})
print submission