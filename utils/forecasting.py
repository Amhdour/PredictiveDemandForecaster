import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .feature_engineering import FeatureEngineer
from .parameter_optimizer import ParameterOptimizer

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

    def prophet_model(self, df, date_column, target_column, selected_features=None, **params):
        """Apply Prophet model with additional regressors"""
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': df[date_column],
            'y': df[target_column]
        })

        # Add selected features as regressors
        if selected_features:
            for feature in selected_features:
                if feature in df.columns:
                    prophet_df[feature] = df[feature]

        # Initialize and fit Prophet model with optimized parameters
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=params.get('seasonality_prior_scale', 10)
        )

        # Add regressors
        if selected_features:
            for feature in selected_features:
                if feature in prophet_df.columns and feature not in ['ds', 'y']:
                    model.add_regressor(feature)

        model.fit(prophet_df)
        return model

    def generate_forecast(self, df, date_column, target_column, model_type,
                         forecast_periods, use_feature_selection=True,
                         optimize_parameters=False, **params):
        """Generate forecast based on selected model"""
        try:
            # Initialize feature engineering if requested
            if use_feature_selection:
                feature_engineer = FeatureEngineer(date_column, target_column)
                enhanced_df, selected_features, importance_scores = feature_engineer.generate_features(df)
                df = enhanced_df
            else:
                selected_features = None

            # Initialize parameter optimization if requested
            if optimize_parameters:
                optimizer = ParameterOptimizer(df, date_column, target_column)

                if model_type == "Moving Average":
                    params = optimizer.optimize_moving_average()
                elif model_type == "Exponential Smoothing":
                    params = optimizer.optimize_exponential_smoothing()
                elif model_type == "ARIMA":
                    params = optimizer.optimize_arima()
                elif model_type == "Prophet":
                    params = optimizer.optimize_prophet()

            data = df[target_column].values
            dates = df[date_column]

            if model_type == "Moving Average":
                # Generate forecast using moving average
                window_size = params.get('window_size', 7)
                ma_values = self.moving_average(df[target_column], window_size)
                forecast = ma_values.iloc[-forecast_periods:].values
                train_pred = ma_values[window_size:].values
                train_actual = data[window_size:]

            elif model_type == "Exponential Smoothing":
                # Generate forecast using exponential smoothing
                alpha = params.get('alpha', 0.3)
                model = self.exponential_smoothing(data, alpha)
                forecast = model.forecast(forecast_periods)
                train_pred = model.fittedvalues
                train_actual = data

            elif model_type == "ARIMA":
                # Generate forecast using ARIMA
                p = params.get('p', 1)
                d = params.get('d', 1)
                q = params.get('q', 1)
                model = self.arima(data, p, d, q)
                forecast = model.forecast(forecast_periods)
                train_pred = model.get_prediction(start=0).predicted_mean
                train_actual = data

            elif model_type == "Prophet":
                # Generate forecast using Prophet with selected features
                model = self.prophet_model(df, date_column, target_column, 
                                        selected_features, **params)

                # Create future dataframe with features
                future_dates = model.make_future_dataframe(periods=forecast_periods)

                # Add feature values for future dates if available
                if selected_features:
                    for feature in selected_features:
                        if feature in df.columns and feature not in ['ds', 'y']:
                            # Use last value for future dates (simple approach)
                            future_dates[feature] = df[feature].iloc[-1]

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

            # Add optimization information if used
            if optimize_parameters:
                metrics['Optimized Parameters'] = str(params)

            # Add feature importance information if available
            if use_feature_selection:
                metrics['Selected Features'] = ', '.join(selected_features)
                metrics['Feature Importance'] = importance_scores.to_dict('records')

            return forecast_df, metrics

        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")