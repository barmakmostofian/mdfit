import os
import argparse
import pandas as pd
import numpy as np
import joblib
import scipy
from scipy.stats import *

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error


# Set up command-line arguments
parser = argparse.ArgumentParser(description='Read data filefrom the command line.')
parser.add_argument('--feat_data',      type=str,   help='Path to the feature data file')
parser.add_argument('--response_data',  type=str,   help='Path to the response data file')
parser.add_argument('--model_type',     type=str,   help='regression model type', help='Use \'lsq\', \'lasso\', or \'ridge\'')
args = parser.parse_args()


# Load data
df_feat     = pd.read_csv(args.feat_data,     sep=',', header=0, index_col=0)
df_response = pd.read_csv(args.response_data, sep=',', header=0, index_col=0)


# Merge data to clearly define feature matrix (X) and response variable (y) for all data instances
# First, redefine mol index column names for correct merging, then remove any instance, for which data are missing 
df_feat.index.name     = 'mol_name'
df_response.index.name = 'mol_name'

df_data = pd.merge(df_feat, df_response, on='mol_name', how='left')

df_data = df_data.dropna()


X = np.array(df_data.drop('potency', axis=1))
y = np.array(df_data['potency'])
ids = np.array(df_data.index)


# UNDER CONSTRUCTION:
# Define regression model 
#if args.model_type == 'lsq' :
#    model = LinearRegression()
#elif args.model_type == 'lasso' :
#    my_alpha_grid = np.linspace(1e-5, 1, 500)
#    model = Lasso(...)
#elif args.model_type == 'ridge' :
#    model = Ridge(...)


my_alpha_grid = np.linspace(1e-5, 1, 500)

# If regularization models are used, we need to apply a nested-CV architecture to optimize hyperparameters

# Define the LOO-CV object for the outer loop, used to split and iterate through the data below
outer_loo = LeaveOneOut()


y_true_all  = [] 
y_pred_all  = [] 
all_best_alphas = []

for train_index, test_index in outer_loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # The true output value of this fold is saved for later comparison with all predicted output values. 
    y_true_all.append(y_test[0])

    # Data are being scaled. 
    # Of course, each training fold on its own. Otherwise, fitting the full dataset before splittinig would leak test statistics into training.
    # The scaler object fitted on the training data is applied to the test data.
    feat_scaler = MinMaxScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled  = feat_scaler.transform(X_test)

    # The inner loop for hyperparameter (alpha) optimization is performed with LassoCV, which efficiently sweeps the alpha grid.
    # The model is initialized, fitted on (scaled) training data and the optimal hyperparameter is saved.
    inner_model = LassoCV(alphas = my_alpha_grid, cv=LeaveOneOut(), max_iter=10_000)
    inner_model.fit(X_train_scaled, y_train)
    best_alpha = inner_model.alpha_
    all_best_alphas.append(best_alpha)

    # The actual (outer) model is trained with optimal hyperparameter alpha.
    outer_model = Lasso(alpha=best_alpha, max_iter=10_000)
    outer_model.fit(X_train_scaled, y_train)

    # The model is now applied and its predicted value is saved to compare to the true value saved above. 
    y_pred = outer_model.predict(X_test_scaled)
    y_pred_all.append(y_pred[0])


print(r2_score(y_true_all, y_pred_all))
print(mean_absolute_error(y_true_all, y_pred_all))
print(kendalltau(y_true_all, y_pred_all)[0])

