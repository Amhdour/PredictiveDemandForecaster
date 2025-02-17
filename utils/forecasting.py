import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
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

    def arima(self, data, p, d, q):
        """Apply ARIMA model"""
        model = SARIMAX(data, order=(p, d, q))
        fitted_model = model.fit(disp=False)
        return fitted_model

    def prophet_model(self, df, date_column, target_column):
        """Apply Prophet model"""
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': df[date_column],
            'y': df[target_column]
        })

        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        return model

    def generate_forecast(self, df, date_column, target_column, model_type,
                         forecast_periods, window_size=None, alpha=None,
                         p=None, d=None, q=None):
        """Generate forecast based on selected model"""
        try:
            data = df[target_column].values
            dates = df[date_column]

            if model_type == "Moving Average":
                # Generate forecast using moving average
                ma_values = self.moving_average(df[target_column], window_size)
                forecast = ma_values.iloc[-forecast_periods:].values
                train_pred = ma_values[window_size:].values
                train_actual = data[window_size:]

            elif model_type == "Exponential Smoothing":
                # Generate forecast using exponential smoothing
                model = self.exponential_smoothing(data, alpha)
                forecast = model.forecast(forecast_periods)
                train_pred = model.fittedvalues
                train_actual = data

            elif model_type == "ARIMA":
                # Generate forecast using ARIMA
                model = self.arima(data, p, d, q)
                forecast = model.forecast(forecast_periods)
                train_pred = model.get_prediction(start=0).predicted_mean
                train_actual = data

            elif model_type == "Prophet":
                # Generate forecast using Prophet
                model = self.prophet_model(df, date_column, target_column)
                future_dates = model.make_future_dataframe(periods=forecast_periods)
                forecast_result = model.predict(future_dates)

                # Extract the forecast values
                forecast = forecast_result.tail(forecast_periods)['yhat'].values
                train_pred = forecast_result['yhat'][:len(data)].values
                train_actual = data

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
            metrics = self.calculate_metrics(train_actual, train_pred)

            return forecast_df, metrics

        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")