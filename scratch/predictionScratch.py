# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:15:03 2022

@author: john.atherfold
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from skopt import BayesSearchCV

dfResults = pd.read_csv('../data/results.csv')
dfScorers = pd.read_csv('../data/goalscorers.csv')
dfShootouts = pd.read_csv('../data/shootouts.csv')

# Formatting and Filtering
dfResults['date'] = pd.to_datetime(dfResults['date'])
nYearsAgo = 10
recentIdx = dfResults['date'].dt.strftime('%Y').astype(int) >= 2022 - nYearsAgo 
dfResultRecent = dfResults[recentIdx]

# Split off current work cup matches
worldCupIdx = dfResultRecent['tournament'] == 'FIFA World Cup'
thisYearIdx = dfResults['date'].dt.strftime('%Y').astype(int) == 2022
inferenceData = dfResults[np.logical_and(worldCupIdx, thisYearIdx)]
trainTestData = dfResults[np.logical_not(np.logical_and(worldCupIdx, thisYearIdx))]

# Create Classification Response
trainTestData['homeWin'] = trainTestData['home_score'] > trainTestData['away_score']
trainTestData['homeLose'] = trainTestData['home_score'] < trainTestData['away_score']
trainTestData['draw'] = trainTestData['home_score'] == trainTestData['away_score']
homeWinSeries = trainTestData['homeWin'].map({True: 'homeWin'})
homeLoseSeries = trainTestData['homeLose'].map({True: 'homeLose'})
drawSeries = trainTestData['draw'].map({True: 'draw'})

trainTestData['results'] = homeWinSeries.astype(str).str.replace('nan','') + homeLoseSeries.astype(str).str.replace('nan','') + drawSeries.astype(str).str.replace('nan','')
trainTestData['resultsNum'] = trainTestData['results'].map({'homeLose':0, 'draw':1, 'homeWin':2})

# Separate into predictors and responses
predictors = trainTestData[['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral']]
inferencePredictors = inferenceData[['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral']]
response = trainTestData['resultsNum']

# One-hot encoding predictors
fullPredictorsDummies = pd.get_dummies(pd.concat([predictors, inferencePredictors]))
predictorsDummies = fullPredictorsDummies[np.logical_not(np.logical_and(worldCupIdx, thisYearIdx))]
inferencePredictorsDummies = fullPredictorsDummies[np.logical_and(worldCupIdx, thisYearIdx)]

# Split into Training and Testing
X_train, X_test, \
    y_train, y_test = train_test_split(predictorsDummies, response, \
                                       test_size=0.20)

# Modelling
mdl = xgb.XGBClassifier()
mdl.fit(X_train, y_train)
yHatTrain = mdl.predict(X_train)
yHatTest = mdl.predict(X_test)

confusion_matrix(y_train, yHatTrain)
confusion_matrix(y_test, yHatTest)

# Optimise Model
cvObj = KFold(n_splits = 10)
param = {
    'xgb_reg__n_estimators': np.arange(100, 250),
    'xgb_reg__reg_lambda': np.logspace(-7, 0, num = 250),
    'xgb_reg__max_depth': np.arange(1, 20),
    'xgb_reg__learning_rate': np.logspace(-3, 0, num = 100),
    'xgb_reg__reg_alpha': np.logspace(-7, 0, num = 250),
    'xgb_reg__subsample': np.arange(0.1, 1.05, 0.05),
    'xgb_reg__colsample_bytree': np.arange(0,1,0.05),
    'xgb_reg__colsample_bylevel': np.arange(0,1,0.05),
    'xgb_reg__colsample_bynode': np.arange(0,1,0.05)
    }

randomSearch = BayesSearchCV(estimator = mdl, search_spaces = param,
                             n_iter = 30, cv = cvObj, verbose = 10, n_jobs = -1,
                             scoring = 'f1')
searchResults = randomSearch.fit(X_train, y_train)

