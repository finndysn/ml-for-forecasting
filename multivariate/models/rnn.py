import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multivariate
from skforecast.model_selection_multiseries import random_search_forecaster_multivariate


def forecast_and_evaluate_knn(df_arg, exog, lag_value, train_size):
    """
    Function to perform time series forecasting using a KNeighborsRegressor,
    optimize hyperparameters using random search, and evaluate the model using backtesting.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index and a single column of time series data.

    Returns:
    dict: Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
    """
    df = df_arg.copy(deep=True)
    df = df.reset_index()
    df = df.drop(df.columns[0], axis=1)

    # Initialize the forecaster with DecisionTreeRegressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=KNeighborsRegressor(),
        level=df.columns[-1], 
        lags=lag_value,
        steps=10, 
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define parameter grid to search for KNeighborsRegressor
    param_grid = {
        'n_neighbors': [3, 5, 10, 20],  # Number of neighbors to use
        'weights': ['uniform', 'distance'],  # Weight function for KNN
        'p': [1, 2]  # Power parameter for the Minkowski metric (1: Manhattan, 2: Euclidean)
    }

    # Perform random search to find the best hyperparameters
    results_random_search = random_search_forecaster_multivariate(
        forecaster=forecaster,
        series=df,  # The column of time series data
        param_distributions=param_grid,
        steps=10,  
        exog=exog,
        n_iter=10,  
        metric='mean_squared_error', 
        initial_train_size=int(len(df) * train_size),  # Use 80% for training, rest for validation
        fixed_train_size=False,  
        return_best=True,  # Return the best parameter set
        random_state=123
    )
    
    best_params = results_random_search.iloc[0]['params']

    # Recreate the forecaster with the best parameters
    forecaster = ForecasterAutoregMultiVariate(
        regressor=KNeighborsRegressor(**best_params),
        level=df.columns[-1], 
        lags=lag_value,
        steps=10, 
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtest the model
    backtest_metric, predictions = backtesting_forecaster_multivariate(
        forecaster=forecaster,
        series=df,
        steps=10,
        metric='mean_squared_error',
        initial_train_size=int(len(df) * train_size),  # 80% train size
        levels=df.columns[-1],   
        exog=exog,
        fixed_train_size=False,  
        verbose=True
    )

    y_true = df.iloc[int(len(df) * train_size):, 0]  # The actual values from the test set
    mae = mean_absolute_error(y_true, predictions)
    mape_val = mean_absolute_percentage_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    # Print evaluation metrics
    print(f"MAE: {mae}")
    print(f"MAPE: {mape_val}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Return results as a dictionary
    return {
        'results_random_search': results_random_search,
        'best_params': best_params,
        'mae': mae,
        'mape': mape_val,
        'mse': mse,
        'rmse': rmse
    }
