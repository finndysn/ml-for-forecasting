import os
import pandas as pd
import numpy as np
import shutil
import csv

from models.random_forest import *
from models.decision_tree import *
from models.elastic_net_regression import *
from models.gradientBoostingRegressor import *
from models.lasso import *
from models.linear_regression import *
from models.ridge import *
from models.rnn import *
from models.svr import *
from models.xgboost import *
from utility.date_functions import *


# we will just use relative path, since the datasets are stored in the same directory as the models.
base_directory = os.path.dirname(__file__)  
# data_directory = os.path.join(base_directory, "univariate", "datasets")
#since we changed directory to the univaraite, we don't need it. 
data_directory = os.path.join(base_directory, "datasets")

# we will only define the size (in %) of the training size
splits = {"split-60-40": 0.6, "split-70-30": 0.7, "split-80-20": 0.8,"split-90-10": 0.9, }

for split_directory, train_size in splits.items():
    eval_directory = os.path.join(base_directory, "evaluations", split_directory)
    os.makedirs(eval_directory, exist_ok=True)
    
    csv_files = [
        ("mae.csv", ["fname","ridge","rf","lr","gb","xbg","dt","lasso","enr","knn"]),
        ("mape.csv", ["fname","ridge","rf","lr","gb","xbg","dt","lasso","enr","knn"]),
        ("mse.csv", ["fname","ridge","rf","lr","gb","xbg","dt","lasso","enr","knn"]),
        ("rmse.csv", ["fname","ridge","rf","lr","gb","xbg","dt","lasso","enr","knn"]),
    ]
    
    for filename, header in csv_files:
        filepath = os.path.join(eval_directory, filename)
        if os.path.exists(filepath):
            print(f"{filepath} already exist. Skip file creation")
        else:
            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                print(f"{filename} created")
    
    #since we changed directory to the univaraite, we don't need it. 
    csv_mae = os.path.join(eval_directory, 'mae.csv')
    csv_mape = os.path.join(eval_directory, 'mape.csv')
    csv_mse = os.path.join(eval_directory, 'mse.csv')
    csv_rmse = os.path.join(eval_directory, 'rmse.csv')

    csv_path = os.path.join(base_directory, "evaluations", split_directory, "mae.csv")
    print(csv_path)
    df_finished_files = pd.read_csv(csv_path)
    list_finished_files = df_finished_files["fname"].to_list()

    for filename in os.listdir(data_directory):
        if filename not in list_finished_files:
            file_path = os.path.join(data_directory, filename)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Step 1: Find the first occurrence of NaN in any column
            first_nan_index = df[df.isna().any(axis=1)].index.min()

            # Step 2: Slice the DataFrame to remove all rows starting from the first NaN
            if pd.notna(first_nan_index):
                df = df.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row

            # freq = infer_frequency(df)
            freq = "D"
            exog = create_time_features(df=df, freq=freq)
            lags = 7

            results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags, train_size=train_size)
            results_rf = forecast_and_evaluate_random_forest(
                df_arg=df, exog=exog, lag_value=lags, train_size=train_size
            )
            results_lr = forecast_and_evaluate_linear_regression(
                df_arg=df, exog=exog, lag_value=lags, train_size=train_size
            )
            results_gb = forecast_and_evaluate_gradient_boosting(
                df_arg=df, exog=exog, lag_value=lags, train_size=train_size
            )
            results_xgb = forecast_and_evaluate_xgboost(df_arg=df, exog=exog, lag_value=lags, train_size=train_size)
            results_dt = forecast_and_evaluate_decision_tree(
                df_arg=df, exog=exog, lag_value=lags, train_size=train_size
            )
            results_lasso = forecast_and_evaluate_lasso(df_arg=df, exog=exog, lag_value=lags, train_size=train_size)
            results_enr = forecast_and_evaluate_elastic_net(
                df_arg=df, exog=exog, lag_value=lags, train_size=train_size
            )
            results_knn = forecast_and_evaluate_knn(df_arg=df, exog=exog, lag_value=lags, train_size=train_size)

            new_row_mae = pd.DataFrame(
                [
                    [
                        filename,
                        results_ridge["mae"],
                        results_rf["mae"],
                        results_lr["mae"],
                        results_gb["mae"],
                        results_xgb["mae"],
                        results_dt["mae"],
                        results_lasso["mae"],
                        results_enr["mae"],
                        results_knn["mae"],
                    ]
                ],
                columns=[
                    "fname",
                    "ridge",
                    "rf",
                    "lr",
                    "gb",
                    "xgb",
                    "dt",
                    "lasso",
                    "enr",
                    "knn",
                ],
            )
            new_row_mape = pd.DataFrame(
                [
                    [
                        filename,
                        results_ridge["mape"],
                        results_rf["mape"],
                        results_lr["mape"],
                        results_gb["mape"],
                        results_xgb["mape"],
                        results_dt["mape"],
                        results_lasso["mape"],
                        results_enr["mape"],
                        results_knn["mape"],
                    ]
                ],
                columns=[
                    "fname",
                    "ridge",
                    "rf",
                    "lr",
                    "gb",
                    "xgb",
                    "dt",
                    "lasso",
                    "enr",
                    "knn",
                ],
            )
            new_row_mse = pd.DataFrame(
                [
                    [
                        filename,
                        results_ridge["mse"],
                        results_rf["mse"],
                        results_lr["mse"],
                        results_gb["mse"],
                        results_xgb["mse"],
                        results_dt["mse"],
                        results_lasso["mse"],
                        results_enr["mse"],
                        results_knn["mse"],
                    ]
                ],
                columns=[
                    "fname",
                    "ridge",
                    "rf",
                    "lr",
                    "gb",
                    "xgb",
                    "dt",
                    "lasso",
                    "enr",
                    "knn",
                ],
            )
            new_row_rmse = pd.DataFrame(
                [
                    [
                        filename,
                        results_ridge["rmse"],
                        results_rf["rmse"],
                        results_lr["rmse"],
                        results_gb["rms
                        results_xgb["rmse"],
                        results_dt["rmse"],
                        results_lasso["rmse"],
                        results_enr["rmse"],
                        results_knn["rmse"],
                    ]
                ],
                columns=[
                    "fname",
                    "ridge",
                    "rf",
                    "lr",
                    "gb",
                    "xgb",
                    "dt",
                    "lasso",
                    "enr",
                    "knn",
                ],
            )

            new_row_mae.to_csv(
                csv_mae, mode="a", header=False, index=False, lineterminator="\n"
            )
            new_row_mape.to_csv(
                csv_mape, mode="a", header=False, index=False, lineterminator="\n"
            )
            new_row_mse.to_csv(
                csv_mse, mode="a", header=False, index=False, lineterminator="\n"
            )
            new_row_rmse.to_csv(
                csv_rmse, mode="a", header=False, index=False, lineterminator="\n"
            )
