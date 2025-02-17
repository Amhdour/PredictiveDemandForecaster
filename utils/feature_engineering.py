import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, date_column, target_column):
        self.date_column = date_column
        self.target_column = target_column
        
    def extract_date_features(self, df):
        """Extract temporal features from date column"""
        date_df = df.copy()
        
        # Extract date components
        date_df['year'] = df[self.date_column].dt.year
        date_df['month'] = df[self.date_column].dt.month
        date_df['day'] = df[self.date_column].dt.day
        date_df['day_of_week'] = df[self.date_column].dt.dayofweek
        date_df['quarter'] = df[self.date_column].dt.quarter
        
        return date_df
        
    def create_lags(self, df, lags=[1, 7, 14, 30]):
        """Create lagged features"""
        lag_df = df.copy()
        
        for lag in lags:
            lag_df[f'lag_{lag}'] = df[self.target_column].shift(lag)
            
        return lag_df
        
    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """Create rolling window features"""
        roll_df = df.copy()
        
        for window in windows:
            roll_df[f'rolling_mean_{window}'] = df[self.target_column].rolling(window=window).mean()
            roll_df[f'rolling_std_{window}'] = df[self.target_column].rolling(window=window).std()
            roll_df[f'rolling_min_{window}'] = df[self.target_column].rolling(window=window).min()
            roll_df[f'rolling_max_{window}'] = df[self.target_column].rolling(window=window).max()
            
        return roll_df
        
    def select_features(self, df, n_features=10):
        """Select most important features using f-regression"""
        # Drop date column and target column
        feature_df = df.drop([self.date_column, self.target_column], axis=1)
        
        # Remove any remaining datetime columns
        feature_df = feature_df.select_dtypes(exclude=['datetime64[ns]'])
        
        # Handle missing values
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X_scaled.shape[1]))
        selector.fit(X_scaled, df[self.target_column].values)
        
        # Get selected feature names
        selected_features = feature_df.columns[selector.get_support()].tolist()
        
        # Calculate feature importance scores
        importance_scores = pd.DataFrame({
            'feature': feature_df.columns,
            'importance': selector.scores_
        }).sort_values('importance', ascending=False)
        
        return selected_features, importance_scores
        
    def generate_features(self, df, include_lags=True, include_rolling=True):
        """Generate all features and select the most important ones"""
        # Extract date features
        enhanced_df = self.extract_date_features(df)
        
        # Add lag features
        if include_lags:
            enhanced_df = self.create_lags(enhanced_df)
            
        # Add rolling features
        if include_rolling:
            enhanced_df = self.create_rolling_features(enhanced_df)
            
        # Select best features
        selected_features, importance_scores = self.select_features(enhanced_df)
        
        return enhanced_df, selected_features, importance_scores
