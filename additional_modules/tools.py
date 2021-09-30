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


def drop_df_columns(df, columns):
    return  df.drop(labels=columns, axis=1, inplace=False).to_numpy()


def get_coefs_of_regression(regression, df):
    coeffs = dict(zip(df.columns.values, np.ravel(regression.coef_)))
    coeffs['intercept'] = regression.intercept_[0]
    return pd.Series(coeffs)


def plot_learning_curves(model, X, y, ax=None):
    if ax is None:
        ax = plt.gca()

    N, train_lc, val_lc = learning_curve(model,
                                         X, y, cv=3,
                                         train_sizes=np.linspace(0.1, 1, 20))
    ax.plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax.plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax.legend()
    



def display_estimator_result(X_train, y_train, C=0.01):
    model = Pipeline([('data_transformer', MinMaxScaler()),
                      ('estimator', LogisticRegression(C=C, fit_intercept=False))])
    cv_result = cross_validate(model, X_train, y_train, cv=5)
    plot_learning_curves(model, X_train, y_train)
    print(cv_result['test_score'])
    return model.fit(X_train, y_train)


def plot_model_accuracy_against_probability(model, X_test, y_test):
    prob = pd.DataFrame(model.predict_proba(X_test))
    pred_true = (y_test == np.argmax(prob.to_numpy(), axis=1))
    sure = (prob[0] - 0.5).abs()
    sure_true_df = pd.DataFrame({'prediction_true': pred_true,
                  'sure': sure})
    fig, axis = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    sns.histplot(x='sure', hue='prediction_true',
                 data=sure_true_df, ax=axis[0],
                 palette=palette, bins=10,
                 multiple='stack')

    sns.kdeplot(
        data=sure_true_df,
        x="sure", hue="prediction_true",
        multiple="fill", clip=(0, sure_true_df['sure'].max()),
        ax = axis[1], palette=palette
    )


def grid_search_learning_curves(generate_model, X, y,
                                grid_param1, grid_param2,
                                param1_name, param2_name):
    'let rows - parameter 1, column - parameter 2'
    # if input values is not in numpy format
    grid_param1 = np.array(grid_param1)
    grid_param2 = np.array(grid_param2)
    # creating subplots
    ax_rows = grid_param1.size
    ax_columns = grid_param2.size
    fig, axes = plt.subplots(ax_rows, ax_columns,
                             figsize=(5*ax_columns, 4*ax_rows))
    # plot work
    for param1_ind, param1_val in enumerate(grid_param1):
        for param2_ind, param2_val in enumerate(grid_param2):
            model = generate_model(param1_val, param2_val)
            ax = axes[param1_ind, param2_ind]
            plot_learning_curves(model, X, y, ax=ax)
    # adding labels for x/y
    for ax, param_val in zip(axes[:,0], grid_param1):
        ylabel = f'{param1_name}={param_val}'
        ax.set_ylabel(ylabel, rotation=0)

    for ax, param_val in zip(axes[0], grid_param2):
        title = f'{param2_name}={param_val}'
        ax.set_title(title)

    fig.tight_layout()
    plt.show()

# class plot_model_pred_accuracy:
#     def __call__(self, model, X_test, y_test):
#         prob = pd.DataFrame(model.predict_proba(X_test))
#         prob['prediction_true'] = y_test == np.argmax(prob.to_numpy(), axis=1)
#
#         results = []
#         for low in np.linspace(0, 0.4, 5):
#             low = round(low, 1)
#             up = round(low + 0.1, 1)
#             err = np.round(self.find_error_in_prob_range(prob, low, up), 3)
#             results.append([low, up, *err])
#         return pd.DataFrame(results)
#
#     @staticmethod
#     def find_error_in_prob_range(df, low, up):
#         deviation = np.abs(df.iloc[:, 0] - 0.5)
#         delta_deviation_mask = ((low < deviation) &
#                                 (up > deviation))
#         mask_size = np.sum(delta_deviation_mask)
#         prediction_results = df.loc[delta_deviation_mask, 'prediction_true']
#         return np.sum(prediction_results) / mask_size, mask_size


