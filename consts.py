from typing import Final

grid_parameters: Final = [
        {
            'estimator__fit_intercept': [True, False],
            'estimator__C': [0.001, 0.01, 0.4, 1, 10, 100],
        }]