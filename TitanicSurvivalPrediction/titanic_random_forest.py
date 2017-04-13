#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 08:12:12 2017

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
from sklearn import cross_validation
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x: len(x))

import re

def get_title(name):
    title_search=re.search(' ([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""
    
titles=titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles==k]=v
          
print(pandas.value_counts(titles))
import operator
family_id_mapping= {}



def get_family_id(row):   
    global family_id_mapping
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name,row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) ==0:
            current_id=1
        else:
            current_id=(max(family_id_mapping.items(),key=operator.itemgetter(1))[1]+1)
        family_id_mapping[family_id]=current_id
    return family_id_mapping[family_id]

#family_id_mapping=get_family_id()

family_ids = titanic.apply(get_family_id,axis=1)

family_ids[titanic["FamilySize"]<3]=-1
          
print(pandas.value_counts(family_ids))

titanic["FamilyId"]=family_ids
titanic["Title"] = titles
family_id_mapping={}

import operator

def get_family_id(row):    
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name,row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) ==0:
            current_id=1
        else:
            current_id=(max(family_id_mapping.items(),key=operator.itemgetter(1))[1]+1)
        family_id_mapping[family_id]=current_id
    return family_id_mapping[family_id]


family_ids = titanic.apply(get_family_id,axis=1)

family_ids[titanic["FamilySize"]<3]=-1
          
print(pandas.value_counts(family_ids))
import matplotlib.pyplot as plt
titanic["FamilyId"]=family_ids

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
# The algorithms we want to ensemble
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
from sklearn.cross_validation import KFold


# Initialize the cross-validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold 
        # We need to use .astype(float) to convert the dataframe to all floats and avoid an sklearn error
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme&#8212;just average the predictions to get the final classification
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
titanic_test=pandas.read_csv("test.csv")
# Put all the predictions together into one array
predictions = np.concatenate(predictions, axis=0)
titles = titanic_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# Check the counts of each unique title
print(pandas.value_counts(titanic_test["Title"]))

# Now we add the family size column
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Now we can add family IDs
# We'll use the same IDs we used earlier
print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"]=titanic_test["Name"].apply( lambda x: len(x))
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })