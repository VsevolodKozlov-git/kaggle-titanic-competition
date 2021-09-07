def get_coefs_of_regression(regression, df):
    return pd.Series(np.ravel(regression.coef_), index=df.columns.values)
