import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')

class ParameterOptimizer:
    def __init__(self, data, date_column, target_column):
        self.data = data
        self.date_column = date_column
        self.target_column = target_column
        
    def create_time_series_splits(self, n_splits=5):
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        y = self.data[self.target_column].values
        return tscv.split(y), y
        
    def optimize_moving_average(self, max_window=30):
        """Optimize window size for Moving Average"""
        splits, y = self.create_time_series_splits()
        
        def objective(window_size):
            errors = []
            window = int(window_size)
            
            for train_idx, val_idx in splits:
                y_train, y_val = y[train_idx], y[val_idx]
                # Calculate moving average on training data
                ma = pd.Series(y_train).rolling(window=window).mean()
                # Use last MA value for predictions
                forecast = np.array([ma.iloc[-1]] * len(y_val))
                # Calculate error
                error = mean_squared_error(y_val, forecast, squared=False)
                errors.append(error)
            
            return np.mean(errors)
        
        # Define search space
        space = hp.quniform('window_size', 1, max_window, 1)
        
        # Optimize
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            show_progressbar=False
        )
        
        return {'window_size': int(best['window_size'])}
        
    def optimize_exponential_smoothing(self):
        """Optimize alpha parameter for Exponential Smoothing"""
        splits, y = self.create_time_series_splits()
        
        def objective(alpha):
            errors = []
            
            for train_idx, val_idx in splits:
                y_train, y_val = y[train_idx], y[val_idx]
                try:
                    # Fit model
                    model = ExponentialSmoothing(y_train)
                    fitted_model = model.fit(smoothing_level=alpha)
                    # Generate forecast
                    forecast = fitted_model.forecast(len(y_val))
                    # Calculate error
                    error = mean_squared_error(y_val, forecast, squared=False)
                    errors.append(error)
                except:
                    return 1e10  # Return large error for failed fits
            
            return np.mean(errors)
        
        # Define search space
        space = hp.uniform('alpha', 0, 1)
        
        # Optimize
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            show_progressbar=False
        )
        
        return {'alpha': float(best['alpha'])}
        
    def optimize_arima(self, max_p=5, max_d=2, max_q=5):
        """Optimize ARIMA parameters"""
        splits, y = self.create_time_series_splits(n_splits=3)  # Fewer splits due to computational intensity
        
        def objective(params):
            p, d, q = int(params['p']), int(params['d']), int(params['q'])
            errors = []
            
            for train_idx, val_idx in splits:
                y_train, y_val = y[train_idx], y[val_idx]
                try:
                    # Fit model
                    model = SARIMAX(y_train, order=(p, d, q))
                    fitted_model = model.fit(disp=False)
                    # Generate forecast
                    forecast = fitted_model.forecast(len(y_val))
                    # Calculate error
                    error = mean_squared_error(y_val, forecast, squared=False)
                    errors.append(error)
                except:
                    return 1e10  # Return large error for failed fits
            
            return np.mean(errors)
        
        # Define search space
        space = {
            'p': hp.quniform('p', 0, max_p, 1),
            'd': hp.quniform('d', 0, max_d, 1),
            'q': hp.quniform('q', 0, max_q, 1)
        }
        
        # Optimize
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            show_progressbar=False
        )
        
        return {
            'p': int(best['p']),
            'd': int(best['d']),
            'q': int(best['q'])
        }
        
    def optimize_prophet(self):
        """Optimize Prophet parameters"""
        splits, y = self.create_time_series_splits(n_splits=3)
        
        def objective(params):
            changepoint_prior_scale = params['changepoint_prior_scale']
            seasonality_prior_scale = params['seasonality_prior_scale']
            errors = []
            
            for train_idx, val_idx in splits:
                train_data = self.data.iloc[train_idx].copy()
                val_data = self.data.iloc[val_idx].copy()
                
                try:
                    # Prepare data for Prophet
                    train_df = pd.DataFrame({
                        'ds': train_data[self.date_column],
                        'y': train_data[self.target_column]
                    })
                    
                    # Fit model
                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )
                    model.fit(train_df)
                    
                    # Generate forecast
                    future_dates = pd.DataFrame({'ds': val_data[self.date_column]})
                    forecast = model.predict(future_dates)
                    
                    # Calculate error
                    error = mean_squared_error(val_data[self.target_column], forecast['yhat'], squared=False)
                    errors.append(error)
                except:
                    return 1e10  # Return large error for failed fits
            
            return np.mean(errors)
        
        # Define search space
        space = {
            'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
            'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10))
        }
        
        # Optimize
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30,  # Fewer evaluations due to computational intensity
            trials=trials,
            show_progressbar=False
        )
        
        return {
            'changepoint_prior_scale': float(best['changepoint_prior_scale']),
            'seasonality_prior_scale': float(best['seasonality_prior_scale'])
        }
