import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Forecaster:
    def calculate_metrics(self, actual, predicted):
        """Calculate performance metrics"""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'R2 Score': round(r2, 2)
        }
        
    def moving_average(self, data, window_size):
        """Calculate moving average forecast"""
        return data.rolling(window=window_size).mean()
        
    def exponential_smoothing(self, data, alpha):
        """Apply exponential smoothing"""
        model = ExponentialSmoothing(data)
        fitted_model = model.fit(smoothing_level=alpha)
        return fitted_model
        
    def sarima(self, data, seasonal_period):
        """Apply SARIMA model"""
        model = SARIMAX(data, order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, seasonal_period))
        fitted_model = model.fit(disp=False)
        return fitted_model
        
    def generate_forecast(self, df, date_column, target_column, model_type,
                         forecast_periods, window_size=None, alpha=None,
                         seasonal_period=None):
        """Generate forecast based on selected model"""
        try:
            data = df[target_column].values
            dates = df[date_column]
            
            if model_type == "Moving Average":
                # Generate forecast using moving average
                ma_values = self.moving_average(df[target_column], window_size)
                forecast = ma_values.iloc[-forecast_periods:].values
                
            elif model_type == "Exponential Smoothing":
                # Generate forecast using exponential smoothing
                model = self.exponential_smoothing(data, alpha)
                forecast = model.forecast(forecast_periods)
                
            elif model_type == "SARIMA":
                # Generate forecast using SARIMA
                model = self.sarima(data, seasonal_period)
                forecast = model.forecast(forecast_periods)
                
            # Create forecast dataframe
            last_date = dates.iloc[-1]
            forecast_dates = pd.date_range(start=last_date, 
                                         periods=forecast_periods + 1,
                                         freq='D')[1:]
            
            forecast_df = pd.DataFrame({
                date_column: forecast_dates,
                'Forecast': forecast
            })
            
            # Calculate metrics on training data
            if model_type == "Moving Average":
                train_pred = ma_values[window_size:].values
                train_actual = data[window_size:]
            elif model_type == "Exponential Smoothing":
                train_pred = model.fittedvalues
                train_actual = data
            else:
                train_pred = model.get_prediction(start=0).predicted_mean
                train_actual = data
                
            metrics = self.calculate_metrics(train_actual, train_pred)
            
            return forecast_df, metrics
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")
