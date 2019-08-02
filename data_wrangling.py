import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def create_feature_matrices(df, ts_len, matrix=None):
    df = df.set_index(['year', 'quarter']).sort_index()

    orig_series, diff_series, rel_diff_series, qdiff_series, rel_qdiff_series = [], [], [], [], []

    for company in df['company'].unique():
        orig = df.loc[df['company'] == company, 'earnings'].rename(company)
        diff = orig - orig.shift(1)
        rel_diff = diff / orig.shift(1)
        qdiff = orig - orig.shift(4)
        rel_qdiff = qdiff / orig.shift(4)

        orig_series.append(orig)
        diff_series.append(diff)
        rel_diff_series.append(rel_diff)
        qdiff_series.append(qdiff)
        rel_qdiff_series.append(rel_qdiff)

    matr = {
        'orig': pd.concat(orig_series, axis=1),
        'diff': pd.concat(diff_series, axis=1),
        'rel_diff': pd.concat(rel_diff_series, axis=1),
        'qdiff': pd.concat(qdiff_series, axis=1),
        'rel_qdiff': pd.concat(rel_qdiff_series, axis=1),
    }

    if not matrix:
        return matr['orig'], matr['diff'], matr['rel_diff'], matr['qdiff'], matr['rel_qdiff']
    else:
        return matr[matrix]


def update_feature_matrices(orig, diff, rel_diff, qdiff, rel_qdiff, y_pred, target, curr_pred_date, companies):
    orig_copy = orig.copy()
    diff_copy = diff.copy()
    rel_diff_copy = rel_diff.copy()
    qdiff_copy = qdiff.copy()
    rel_qdiff_copy = rel_qdiff.copy()

    prev_1_pred_date = curr_pred_date.get_prev(n_quarters=1)
    prev_4_pred_date = curr_pred_date.get_prev(n_quarters=4)

    prev_1_orig = orig_copy.loc[(prev_1_pred_date.year, prev_1_pred_date.quarter), companies]
    prev_4_orig = orig_copy.loc[(prev_4_pred_date.year, prev_4_pred_date.quarter), companies]

    if target == '0_orig':
        orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = y_pred
    elif target == '0_diff':
        orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = prev_1_orig + y_pred
    elif target == '0_rel_diff':
        orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = prev_1_orig + (prev_1_orig * y_pred)
    elif target == '0_qdiff':
        orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = prev_4_orig + y_pred
    elif target == '0_rel_qdiff':
        orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = prev_4_orig + (prev_4_orig * y_pred)

    curr_orig = orig_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies]

    diff_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = curr_orig - prev_1_orig
    rel_diff_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = (curr_orig - prev_1_orig) / prev_1_orig
    qdiff_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = curr_orig - prev_4_orig
    rel_qdiff_copy.loc[(curr_pred_date.year, curr_pred_date.quarter), companies] = (curr_orig - prev_4_orig) / prev_4_orig

    return orig_copy, diff_copy, rel_diff_copy, qdiff_copy, rel_qdiff_copy


def get_feature_ranges(ts_len):
    orig_range = [f for f in range(0, ts_len + 1)]
    diff_range = [f for f in range(0, ts_len)]
    rel_diff_range = [f for f in range(0, ts_len)]
    qdiff_range = [f for f in range(0, ts_len - 3)]
    rel_qdiff_range = [f for f in range(0, ts_len - 3)]

    return orig_range, diff_range, rel_diff_range, qdiff_range, rel_qdiff_range


def get_feature_names(orig_range, diff_range, rel_diff_range, qdiff_range, rel_qdiff_range):
    orig_names = [f'{f}_orig' for f in orig_range]
    diff_names = [f'{f}_diff' for f in diff_range]
    rel_diff_names = [f'{f}_rel_diff' for f in rel_diff_range]
    qdiff_names = [f'{f}_qdiff' for f in qdiff_range]
    rel_qdiff_names = [f'{f}_rel_qdiff' for f in rel_qdiff_range]

    return orig_names, diff_names, rel_diff_names, qdiff_names, rel_qdiff_names


def get_feature_name_mappers(orig_range, diff_range, rel_diff_range, qdiff_range, rel_qdiff_range, orig_names,
                             diff_names, rel_diff_names, qdiff_names, rel_qdiff_names):
    orig_mapper = dict(zip(orig_range, orig_names))
    diff_mapper = dict(zip(diff_range, diff_names))
    rel_diff_mapper = dict(zip(rel_diff_range, rel_diff_names))
    qdiff_mapper = dict(zip(qdiff_range, qdiff_names))
    rel_qdiff_mapper = dict(zip(rel_qdiff_range, rel_qdiff_names))

    return orig_mapper, diff_mapper, rel_diff_mapper, qdiff_mapper, rel_qdiff_mapper


def get_feature_dates(orig_range, diff_range, rel_diff_range, qdiff_range, rel_qdiff_range, y_train_date):
    orig_feature_dates = [(y_train_date.get_prev(f).year, y_train_date.get_prev(f).quarter) for f in orig_range]
    diff_feature_dates = [(y_train_date.get_prev(f).year, y_train_date.get_prev(f).quarter) for f in diff_range]
    rel_diff_feature_dates = [(y_train_date.get_prev(f).year, y_train_date.get_prev(f).quarter) for f in rel_diff_range]
    qdiff_feature_dates = [(y_train_date.get_prev(f).year, y_train_date.get_prev(f).quarter) for f in qdiff_range]
    rel_qdiff_feature_dates = [(y_train_date.get_prev(f).year, y_train_date.get_prev(f).quarter) for f in rel_qdiff_range]

    return orig_feature_dates, diff_feature_dates, rel_diff_feature_dates, qdiff_feature_dates, rel_qdiff_feature_dates


class ErrCollection:
    def __init__(self):
        self._rows = []

    def add(self, company, ts_len, pred_steps, year, quarter, err):
        self._rows.append({
            'company': company,
            'ts_len': ts_len,
            'pred_steps': pred_steps,
            'year': year,
            'quarter': quarter,
            'err': err
        })

    def get_per_company_err(self, err_name):
        df = pd.DataFrame(self._rows).rename(columns={'err': err_name})
        df = df.groupby(['company', 'ts_len', 'pred_steps', 'year', 'quarter']).mean()
        df.loc[df[err_name] > 1, err_name] = 1

        df = df.groupby(['ts_len', 'pred_steps', 'company']).mean()
        return df[err_name]


def display_err_stats(br_errs, qep_errs):
    errs = pd.concat([qep_errs, br_errs], axis=1).groupby(['ts_len', 'pred_steps'])

    ttest_p = errs.apply(lambda x: ttest_rel(x['br_mape'], x['qep_mape']).pvalue / 2).rename('ttest_p')
    wilcoxon_p = errs.apply(lambda x: wilcoxon(x['br_mape'], x['qep_mape']).pvalue / 2).rename('wilcoxon_p')

    qep_mean = errs.apply(lambda x: x['qep_mape'].mean()).rename('qep_mean')
    br_mean = errs.apply(lambda x: x['br_mape'].mean()).rename('br_mean')

    print(pd.concat([ttest_p, wilcoxon_p, qep_mean, br_mean], axis=1))
    p_values = pd.concat([ttest_p, wilcoxon_p], axis=0)
    print('proportion <=0.05: ', p_values[p_values <= 0.05].count() / p_values.count())
    print()
