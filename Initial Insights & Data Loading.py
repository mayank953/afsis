# =============================================================================
# Loading the Data & libraries
# =============================================================================
import os
import pandas as pd

import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestRegressor

os.chdir("C:/Users/mayan/Downloads/CHERUVU/KAGGLE")

train= pd.read_csv("training.csv")
test = pd.read_csv("sorted_test.csv")

# =============================================================================
# EDA  
# =============================================================================
train.shape
train.head(5)

test.shape
test.head(5)
labels = train[['Ca','P','pH','SOC','Sand']].values


#Dropping the Dependent Variable & PIDN as they Won't be used For Training or in Preprocessing
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
# now both Test & Train have 727 x 3594 & 1157 x 3594 shape respectively

# missing Value & its treatment

train.columns[train.isnull().any()]
test.columns[train.isnull().any()]
# as we can see no missing Values are present here so we can proceed 

train.dtypes
test.dtypes
##train.groupby('Depth').count()
train.groupby('Depth').size()
test.groupby('Depth').size()
# Here we can see the Distribution of our Dpeth Variable.

# as we can see out of all variables only Depth is the one with Categories

train.groupby('Depth').count()
train.groupby('Depth').size()

#converting the Train & test Depth Variable to Float with Topsoil = 1 & subsoil =0
train['Depth1'] = train['Depth'].map(lambda x: 1 if x == 'Topsoil' else 0).astype(float)
test['Depth1'] = test['Depth'].map(lambda x: 1 if x == 'Topsoil' else 0).astype(float)


train.drop('Depth', axis=1, inplace=True)
test.drop('Depth', axis=1, inplace=True)

train.describe()
test.describe()


# =============================================================================
# optional Processing
# =============================================================================



# =============================================================================
# Dropping Depth & Dropping GeologicalFeatures if Required
# =============================================================================

train.drop(['BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD', 'LSTN', 'REF1', 'REF2', 'REF3', 'REF7', 'RELI', 'TMAP', 'TMFI', 'Depth'], axis=1, inplace=True)
test.drop(['BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD', 'LSTN', 'REF1', 'REF2', 'REF3', 'REF7', 'RELI', 'TMAP', 'TMFI', 'Depth'], axis=1, inplace=True)
# =============================================================================
# CO2 Column Variable - advised to be Dropped
# =============================================================================

train.drop(["m2379.76", "m2377.83", "m2375.9",  "m2373.97", "m2372.04", "m2370.11", "m2368.18", "m2366.26", "m2364.33", "m2362.4",  "m2360.47", "m2358.54", "m2356.61", "m2354.68", "m2352.76"], axis=1, inplace=True)
test.drop(["m2379.76", "m2377.83", "m2375.9",  "m2373.97", "m2372.04", "m2370.11", "m2368.18", "m2366.26", "m2364.33", "m2362.4",  "m2360.47", "m2358.54", "m2356.61", "m2354.68", "m2352.76"], axis=1, inplace=True)
column_list = list(train.columns.values)