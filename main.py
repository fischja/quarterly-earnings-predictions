import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from br_arima import get_br_arima_errs
from qep_model import get_qep_errs
from data_wrangling import display_err_stats

pd.set_option('display.float_format', '{:.4f}'.format)
np.random.seed(42)

earnings_path = r'.\data\earnings.csv'
earnings = pd.read_csv(earnings_path, header=0, usecols=['company', 'year', 'quarter', 'earnings'], thousands=",")

# validation errors
val_pred_year = 2016
br_val_errs = get_br_arima_errs(earnings=earnings, pred_year=val_pred_year)
qep_val_errs = get_qep_errs(earnings=earnings, pred_year=val_pred_year)
display_err_stats(br_errs=br_val_errs, qep_errs=qep_val_errs)

# testing errors
test_pred_year = 2017
br_test_errs = get_br_arima_errs(earnings=earnings, pred_year=test_pred_year)
qep_test_errs = get_qep_errs(earnings=earnings, pred_year=test_pred_year)
display_err_stats(br_errs=br_test_errs, qep_errs=qep_test_errs)
