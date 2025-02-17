import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def plot_time_series(self, df, date_column, target_column):
        """Create time series plot"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[target_column],
                name='Historical Data',
                line=dict(color='#1f77b4')
            )
        )
        
        fig.update_layout(
            title='Historical Time Series Data',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
        
    def plot_forecast(self, historical_df, forecast_df, date_column, target_column):
        """Create forecast plot"""
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df[date_column],
                y=historical_df[target_column],
                name='Historical Data',
                line=dict(color='#1f77b4')
            )
        )
        
        # Plot forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_df[date_column],
                y=forecast_df['Forecast'],
                name='Forecast',
                line=dict(color='#ff7f0e')
            )
        )
        
        fig.update_layout(
            title='Forecast Results',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
