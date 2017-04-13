#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:39:04 2017

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
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg=LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

titanic_test=pandas.read_csv("test.csv")

titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"]=2
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())

alg=LogisticRegression(random_state=1)

alg.fit(titanic[predictors],titanic["Survived"])
predictions=alg.predict(titanic_test[predictors])

submission =pandas.DataFrame({"PassengerId":titanic_test["PassengerId"],"Survived":predictions})
submission.to_csv("submission.csv", index= False)