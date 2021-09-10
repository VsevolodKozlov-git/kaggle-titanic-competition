# basic ml imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# applying beatuful styling 
import seaborn as sns
sns.set()
# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
# constants
import consts

def drop_df_columns(df, columns):
    return  df.drop(labels=columns, axis=1, inplace=False).to_numpy()


def get_coefs_of_regression(regression, df):
    return pd.Series(np.ravel(regression.coef_), index=df.columns.values)


def plot_learning_curves(model, X, y):
    N, train_lc, val_lc = learning_curve(model,
                                         X, y, cv=3,
                                         train_sizes=np.linspace(0.1, 1, 20))
    plt.plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    plt.plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    plt.legend()
    

def train_estimator_via_grid(X, y):
    """
    Train estimator.
    Print: score, best parameters.
    Plot: learning curves
    :return: best pipeline, X test, y test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    pipeline = Pipeline([('data_transformer', MinMaxScaler()), ('estimator', LogisticRegression())])
    grid = GridSearchCV(pipeline, consts.grid_parameters, cv=5)
    grid.fit(X_train, y_train)
    plot_learning_curves(grid.best_estimator_, X, y)
    print(f'score: {grid.score(X_test, y_test)}')
    print(f'best params: {grid.best_params_}')
    return grid.best_estimator_, X_test, y_test

def display_estimator_result(X_train, y_train, C=0.01):
    model = Pipeline([('data_transformer', MinMaxScaler()),
                      ('estimator', LogisticRegression(C=C, fit_intercept=False))])
    cv_result = cross_validate(model, X_train, y_train, cv=5)
    plot_learning_curves(model, X_train, y_train)
    print(cv_result['test_score'])
    return model.fit(X_train, y_train)