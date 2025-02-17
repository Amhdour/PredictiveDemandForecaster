import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, file=None):
        self.file = file

    def load_data(self):
        """Load data from uploaded file"""
        try:
            if self.file is None:
                return self.generate_sample_data()

            if self.file.name.endswith('.csv'):
                df = pd.read_csv(self.file)
            else:
                df = pd.read_excel(self.file)
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    def generate_sample_data(self):
        """Generate sample time series data"""
        np.random.seed(42)

        # Generate dates for the past year
        dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq='D')

        # Generate components
        trend = np.linspace(100, 200, len(dates))  # Upward trend
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Yearly seasonality
        weekly = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 10, len(dates))  # Random noise

        # Combine components
        demand = trend + seasonal + weekly + noise

        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Demand': demand.round(2),
            'Category': np.random.choice(['A', 'B', 'C'], size=len(dates))
        })

        return df

    def preprocess_data(self, df, date_column, target_column, missing_handler):
        """Preprocess the data based on user selections"""
        try:
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column])

            # Sort by date
            df = df.sort_values(date_column)

            # Handle missing values
            if missing_handler == "Forward Fill":
                df[target_column] = df[target_column].fillna(method='ffill')
            elif missing_handler == "Backward Fill":
                df[target_column] = df[target_column].fillna(method='bfill')
            elif missing_handler == "Linear Interpolation":
                df[target_column] = df[target_column].interpolate(method='linear')

            # Remove any remaining missing values
            df = df.dropna(subset=[target_column])

            return df

        except Exception as e:
            raise Exception(f"Error preprocessing data: {str(e)}")