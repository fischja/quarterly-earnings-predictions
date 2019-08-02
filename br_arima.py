import numpy as np
import pandas as pd
from scipy.optimize import minimize
from date_utilities import YearQuarter


class BrArima:
    def __init__(self):
        self._alphas = None
        self._betas = None

    def fit_predict(self, X, steps):
        self._fit(X=X)
        return self._predict(X=X, steps=steps)

    def _fit(self, X):
        X = np.atleast_2d(X)
        alphas = np.zeros(X.shape[0])
        betas = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            res = minimize(fun=lambda p: self._mean_squared_residual(ts=X[i], alpha=p[0], beta=p[1]),
                           x0=[0.5, 0.5],  method='CG', tol=1e-6, options={'eps': 0.02, 'maxiter': 100000})
            alphas[i] = res.x[0]
            betas[i] = res.x[1]
        self._alphas = alphas
        self._betas = betas

    def _predict(self, X, steps):
        X = np.atleast_2d(X)
        residuals = np.zeros(shape=(X.shape[0], X.shape[1] + steps))
        residuals[:, :-steps] = self._get_residuals(ts=X, alpha=self._alphas, beta=self._betas)

        predictions = np.zeros(shape=(X.shape[0], X.shape[1] + steps))
        predictions[:, :-steps] = X.copy()

        for i in range(X.shape[1], X.shape[1] + steps):
            predictions[:, i] = (predictions[:, i-4]
                                 + (self._alphas * (predictions[:, i-1] - predictions[:, i-5]))
                                 + residuals[:, i]
                                 - (self._betas * residuals[:, i-4]))
        return predictions[:, -steps:]

    def _mean_squared_residual(self, ts, alpha, beta):
        residuals = self._get_residuals(ts=ts, alpha=alpha, beta=beta)
        return np.mean(np.square(residuals))

    def _get_residuals(self, ts, alpha, beta):
        ts = np.atleast_2d(ts)
        residuals = np.zeros(ts.shape)
        for i in range(5, ts.shape[1]):
            residuals[:, i] = (ts[:, i] - ts[:, i-4]
                               - (alpha * (ts[:, i-1] - ts[:, i-5]))
                               + (beta * residuals[:, i-4]))
        return residuals


def get_br_arima_errs(earnings, pred_year, min_ts_len=6, max_ts_len=12, max_pred_steps=4):
    earnings = earnings.sort_values(by=['company', 'year', 'quarter'], ascending=True).set_index(['year', 'quarter'])

    ts = [earnings.loc[earnings['company'] == company, 'earnings'].rename(company)
          for company in earnings['company'].unique()]
    df = pd.concat(ts, axis=1)

    FIRST_PRED_DATE = YearQuarter(year=pred_year, quarter=1)
    LAST_PRED_DATE = YearQuarter(year=pred_year, quarter=4)
    max_pred_steps = min(max_pred_steps, FIRST_PRED_DATE.abs_quarters_diff(LAST_PRED_DATE) + 1)

    res = []
    for ts_len in range(min_ts_len, max_ts_len + 1):
        for curr_max_pred_steps in range(max_pred_steps, 0, -1):
            first_pred_date = LAST_PRED_DATE.get_prev(n_quarters=curr_max_pred_steps-1)
            last_train_date = first_pred_date.get_prev(n_quarters=1)
            first_train_date = last_train_date.get_prev(n_quarters=ts_len-1)

            first_train_date_idx = df.index.get_loc((first_train_date.year, first_train_date.quarter))
            last_train_date_idx = df.index.get_loc((last_train_date.year, last_train_date.quarter))
            first_pred_date_idx = df.index.get_loc((first_pred_date.year, first_pred_date.quarter))
            last_pred_date_idx = df.index.get_loc((LAST_PRED_DATE.year, LAST_PRED_DATE.quarter))

            X = df.iloc[first_train_date_idx : last_train_date_idx + 1].T
            br = BrArima()
            predicted = br.fit_predict(X=X, steps=curr_max_pred_steps).T
            correct = df.iloc[first_pred_date_idx: last_pred_date_idx + 1]

            abs_perc_errs = abs((correct - predicted) / correct)
            abs_perc_errs = abs_perc_errs.assign(pred_steps=range(1, curr_max_pred_steps + 1),
                                                 ts_len=ts_len)
            abs_perc_errs = abs_perc_errs.set_index(append=True, keys=['ts_len', 'pred_steps'])

            for c in abs_perc_errs.columns:
                res.append(abs_perc_errs[[c]].assign(company=c).rename(columns={c: 'br_mape'}).reset_index())

    df_res = pd.concat(res, axis=0).reset_index(drop=True)
    df_res = df_res.groupby(['company', 'ts_len', 'pred_steps', 'year', 'quarter']).mean()
    df_res[df_res > 1] = 1

    return df_res.groupby(['ts_len', 'pred_steps', 'company']).mean()
