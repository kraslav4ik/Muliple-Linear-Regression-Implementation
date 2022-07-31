import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class TrainModelException(Exception):
    def __init__(self):
        self.message = 'Fit model with data first'
        super().__init__(self.message)


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.coefficients = np.array([])
        self.intercept = 0

    def independent_to_array(self, independent: pd.DataFrame):
        X = independent.copy(deep=True)
        if self.fit_intercept:
            ones = [1 for _ in range(X.shape[0])]
            X.insert(0, "ones", ones)
        return X.to_numpy()

    def dependent_to_array(self, dependent: pd.DataFrame):
        return dependent.to_numpy()

    def fit(self, independent: pd.DataFrame, dependent: pd.DataFrame) -> None:
        X, y = self.independent_to_array(independent), self.dependent_to_array(dependent)
        X_transposed = X.transpose()
        first_prod = np.matmul(X_transposed, X)
        inv = linalg.inv(first_prod)
        second_prod = np.matmul(inv, X_transposed)
        res_array = np.matmul(second_prod, y)
        self.coefficients = np.array(res_array).transpose()[0]
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
        return

    def predict(self, independent: pd.DataFrame) -> np.array:
        if not self.coefficients.size:
            raise TrainModelException
        X = self.independent_to_array(independent)
        return (X @ self.coefficients.transpose())[np.newaxis].transpose()

    def r2_score(self, dependent: pd.DataFrame, yhat: np.array) -> float:
        y = self.dependent_to_array(dependent)
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def rmse(self, dependent, yhat) -> float:
        y = self.dependent_to_array(dependent)
        return np.sqrt(np.sum((y - yhat) ** 2) / yhat.shape[0])


def main() -> None:
    cols = pd.read_csv('./data.csv', nrows=1).columns.tolist()
    table = pd.read_csv('./data.csv', skiprows=1, names=cols[:-1] + ['y'])
    dep = pd.DataFrame(table['y'])
    indep = table.drop('y', axis=1)
    print('Fit model with intercept? - "y" or "n"')
    intercept = input()
    if intercept not in {'y', 'n'}:
        raise ValueError(intercept)
    fit_intercept = True if intercept == 'y' else False

    sci_lg = LinearRegression(fit_intercept=fit_intercept)
    sci_lg.fit(indep, dep)
    sci_predictions = sci_lg.predict(indep)
    sci_r2_score = r2_score(dep, sci_predictions)
    sci_rmse = mean_squared_error(dep, sci_predictions) ** 0.5

    custom_lg = CustomLinearRegression(fit_intercept=fit_intercept)
    custom_lg.fit(indep, dep)
    yhat = custom_lg.predict(indep)
    custom_r2_score = custom_lg.r2_score(dep, yhat)
    custom_rmse = custom_lg.rmse(dep, yhat)
    print({'intercept': abs((sci_lg.intercept_[0] if sci_lg.intercept_ else 0) - custom_lg.intercept),
           'Coefficient': abs(sci_lg.coef_[0] -
                              (custom_lg.coefficients[1:] if custom_lg.intercept else custom_lg.coefficients)),
           'R2': abs(sci_r2_score - custom_r2_score), 'RMSE': abs(sci_rmse - custom_rmse)})

    sci_pred = sci_predictions.transpose()[0]
    custom_pred = yhat.transpose()[0]
    fig, ax = plt.subplots()
    fig.suptitle(f'Comparison of predictions by SkLearn Linear regression and My linear regression model\n'
                 f'Intercept = {fit_intercept}')
    fig.set_size_inches(10, 6)
    p1 = max(max(sci_pred), max(dep['y']))
    p2 = min(min(sci_pred), min(dep['y']))
    ax.scatter(dep, sci_pred, c='g', alpha=0.5, s=35, label='SkLearn linear regression predictions')
    ax.plot([p1, p2], [p1, p2], 'y')
    ax.scatter(dep, custom_pred, c='crimson', alpha=0.8, s=5, label='My linear regression predictions')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Predictions')
    ax.set_xlabel('True Values')
    ax.set_aspect('equal', adjustable='box')


    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.show()


if __name__ == "__main__":
    main()
