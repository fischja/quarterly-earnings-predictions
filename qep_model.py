import itertools
import numpy as np
import pandas as pd
from date_utilities import YearQuarter
from data_wrangling import (
    ErrCollection,
    get_feature_dates,
    get_feature_name_mappers,
    create_feature_matrices,
    get_feature_names,
    get_feature_ranges,
    update_feature_matrices
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR


def get_qep_errs(earnings, pred_year, min_ts_len=6, max_ts_len=12, max_pred_steps=4):
    mape_errs = ErrCollection()

    for ts_len in range(min_ts_len, max_ts_len + 1):
        print()
        print(f'ts_len:{ts_len}')
        chosen_features = []

        orig, diff, rel_diff, qdiff, rel_qdiff = create_feature_matrices(earnings, ts_len)

        orig_range, diff_range, rel_diff_range, qdiff_range, rel_qdiff_range = get_feature_ranges(ts_len)

        orig_names, diff_names, rel_diff_names, qdiff_names, rel_qdiff_names = get_feature_names(
            orig_range=orig_range,
            diff_range=diff_range,
            rel_diff_range=rel_diff_range,
            qdiff_range=qdiff_range,
            rel_qdiff_range=rel_qdiff_range)

        orig_mapper, diff_mapper, rel_diff_mapper, qdiff_mapper, rel_qdiff_mapper = get_feature_name_mappers(
            orig_range=orig_range,
            diff_range=diff_range,
            rel_diff_range=rel_diff_range,
            qdiff_range=qdiff_range,
            rel_qdiff_range=rel_qdiff_range,
            orig_names=orig_names,
            diff_names=diff_names,
            rel_diff_names=rel_diff_names,
            qdiff_names=qdiff_names,
            rel_qdiff_names=rel_qdiff_names)

        companies = np.copy(orig.columns.values)
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)

        for train_company_idxs, val_company_idxs in cv.split(companies):
            train_companies = companies[train_company_idxs]
            val_companies = companies[val_company_idxs]

            start_pred_date = YearQuarter(pred_year, 1)
            max_pred_date = YearQuarter(pred_year, 4)

            while True:
                orig_temp = orig[val_companies].copy()
                diff_temp = diff[val_companies].copy()
                rel_diff_temp = rel_diff[val_companies].copy()
                qdiff_temp = qdiff[val_companies].copy()
                rel_qdiff_temp = rel_qdiff[val_companies].copy()

                y_train_date = YearQuarter(2012, 1).get_next(ts_len)
                train = []

                while True:
                    orig_dates, diff_dates, rel_diff_dates, qdiff_dates, rel_qdiff_dates = get_feature_dates(
                        orig_range=orig_range,
                        diff_range=diff_range,
                        rel_diff_range=rel_diff_range,
                        qdiff_range=qdiff_range,
                        rel_qdiff_range=rel_qdiff_range,
                        y_train_date=y_train_date)

                    orig_train = orig.loc[orig_dates, train_companies].reset_index(drop=True).rename(index=orig_mapper)
                    diff_train = diff.loc[diff_dates, train_companies].reset_index(drop=True).rename(index=diff_mapper)
                    rel_diff_train = rel_diff.loc[rel_diff_dates, train_companies].reset_index(drop=True).rename(index=rel_diff_mapper)
                    qdiff_train = qdiff.loc[qdiff_dates, train_companies].reset_index(drop=True).rename(index=qdiff_mapper)
                    rel_qdiff_train = rel_qdiff.loc[rel_qdiff_dates, train_companies].reset_index(drop=True).rename(index=rel_qdiff_mapper)

                    train.append(pd.concat([orig_train, diff_train, rel_diff_train, qdiff_train, rel_qdiff_train], sort=False).T)

                    y_train_date = y_train_date.get_next()
                    if y_train_date.equals(start_pred_date):
                        break

                train = pd.concat(train)

                poss_y_cols = [f for f in train.columns if f.startswith('0')]
                x_cols = [f for f in np.setdiff1d(train.columns, poss_y_cols) if 'rel' not in f]
                y_col = '0_qdiff'

                x_train = train[x_cols]
                y_train = train[y_col].values.reshape(-1,)

                # feature selection
                feature_selector = SelectKBest(mutual_info_regression, k=4).fit(x_train, y_train)

                feature_mask = feature_selector.get_support()
                chosen_features.append(x_train.columns.values[feature_mask])

                x_train = feature_selector.transform(x_train)

                # feature scaling
                feature_scalar = QuantileTransformer(n_quantiles=min(1000, x_train.shape[0]),
                                                     output_distribution='normal',
                                                     random_state=42).fit(x_train)
                x_train = feature_scalar.transform(x_train)

                # target scaling
                target_scalar = QuantileTransformer(n_quantiles=min(1000, y_train.shape[0]),
                                                    output_distribution='normal',
                                                    random_state=42).fit(y_train.reshape(-1, 1))
                y_train = target_scalar.transform(y_train.reshape(-1, 1)).reshape(-1,)

                model = SVR(epsilon=0.04, C=0.2, kernel='rbf', gamma=0.25).fit(x_train, y_train)

                total_pred_steps = min(max_pred_steps, start_pred_date.abs_quarters_diff(max_pred_date) + 1)
                curr_pred_date = start_pred_date.clone()

                for pred_steps in range(1, total_pred_steps + 1):  # make rolling predictions
                    orig_dates, diff_dates, rel_diff_dates, qdiff_dates, rel_qdiff_dates = get_feature_dates(
                        orig_range=orig_range,
                        diff_range=diff_range,
                        rel_diff_range=rel_diff_range,
                        qdiff_range=qdiff_range,
                        rel_qdiff_range=rel_qdiff_range,
                        y_train_date=curr_pred_date)

                    orig_val = orig_temp.loc[orig_dates].reset_index(drop=True).rename(index=orig_mapper)
                    diff_val = diff_temp.loc[diff_dates].reset_index(drop=True).rename(index=diff_mapper)
                    rel_diff_val = rel_diff_temp.loc[rel_diff_dates].reset_index(drop=True).rename(index=rel_diff_mapper)
                    qdiff_val = qdiff_temp.loc[qdiff_dates].reset_index(drop=True).rename(index=qdiff_mapper)
                    rel_qdiff_val = rel_qdiff_temp.loc[rel_qdiff_dates].reset_index(drop=True).rename(index=rel_qdiff_mapper)

                    val = pd.concat([orig_val, diff_val, rel_diff_val, qdiff_val, rel_qdiff_val], sort=False).T

                    x_val = val[x_cols]
                    y_val = val[y_col]

                    x_val = feature_selector.transform(x_val)
                    x_val = feature_scalar.transform(x_val)
                    y_pred = model.predict(x_val).reshape(-1, 1)
                    y_pred = target_scalar.inverse_transform(y_pred).reshape(-1,)

                    orig_temp, diff_temp, rel_diff_temp, qdiff_temp, rel_qdiff_temp = update_feature_matrices(
                        orig=orig_temp,
                        diff=diff_temp,
                        rel_diff=rel_diff_temp,
                        qdiff=qdiff_temp,
                        rel_qdiff=rel_qdiff_temp,
                        y_pred=y_pred,
                        target=y_val.name,
                        curr_pred_date=curr_pred_date,
                        companies=val_companies)

                    for company in val_companies:
                        actual = orig.loc[(curr_pred_date.year, curr_pred_date.quarter), company]
                        forecast = orig_temp.loc[(curr_pred_date.year, curr_pred_date.quarter), company]

                        ape = abs((actual - forecast) / actual)
                        mape_errs.add(company=company,
                                      ts_len=ts_len,
                                      pred_steps=pred_steps,
                                      year=curr_pred_date.year,
                                      quarter=curr_pred_date.quarter,
                                      err=ape)

                    curr_pred_date = curr_pred_date.get_next()

                if start_pred_date.equals(max_pred_date):
                    break
                else:
                    start_pred_date = start_pred_date.get_next()

        feature_importances = pd.DataFrame(
            {ts_len: list(itertools.chain.from_iterable(chosen_features))}
        ).iloc[:, 0].value_counts()
        feature_importances = ((feature_importances / 40) * 100)
        print('feature_importances')
        print(feature_importances)

    return mape_errs.get_per_company_err(err_name=f'qep_mape')
