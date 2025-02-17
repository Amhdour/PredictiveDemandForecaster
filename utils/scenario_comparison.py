import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ForecastScenario:
    name: str
    model_type: str
    parameters: Dict[str, Any]
    forecast_df: pd.DataFrame
    metrics: Dict[str, Any]

class ScenarioManager:
    def __init__(self):
        self.scenarios = {}

    def add_scenario(self, name: str, model_type: str, parameters: Dict[str, Any],
                    forecast_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Add a new forecast scenario"""
        self.scenarios[name] = ForecastScenario(
            name=name,
            model_type=model_type,
            parameters=parameters,
            forecast_df=forecast_df,
            metrics=metrics
        )

    def get_scenario_names(self) -> List[str]:
        """Get list of available scenarios"""
        return list(self.scenarios.keys())

    def get_scenario(self, name: str) -> ForecastScenario:
        """Get a specific scenario"""
        return self.scenarios.get(name)

    def clear_scenarios(self):
        """Clear all scenarios"""
        self.scenarios = {}

    def plot_comparison(self, historical_df: pd.DataFrame, 
                       date_column: str, target_column: str,
                       selected_scenarios: List[str] = None):
        """Create comparison plot for selected scenarios"""
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

        # Plot each selected scenario
        if selected_scenarios is None:
            selected_scenarios = self.get_scenario_names()

        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, scenario_name in enumerate(selected_scenarios):
            scenario = self.get_scenario(scenario_name)
            if scenario:
                fig.add_trace(
                    go.Scatter(
                        x=scenario.forecast_df[date_column],
                        y=scenario.forecast_df['Forecast'],
                        name=f'{scenario.name} ({scenario.model_type})',
                        line=dict(color=colors[i % len(colors)])
                    )
                )

        fig.update_layout(
            title='Forecast Scenario Comparison',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def compare_metrics(self) -> pd.DataFrame:
        """Compare metrics across all scenarios"""
        metrics_data = []

        for name, scenario in self.scenarios.items():
            # Extract basic metrics
            scenario_metrics = {
                'Scenario': name,
                'Model': scenario.model_type
            }

            # Add model parameters (handle both optimized and manual parameters)
            if isinstance(scenario.parameters, dict):
                for param_name, param_value in scenario.parameters.items():
                    scenario_metrics[f'Param_{param_name}'] = param_value

            # Add performance metrics
            for metric_name, metric_value in scenario.metrics.items():
                if isinstance(metric_value, (int, float)):
                    scenario_metrics[metric_name] = metric_value
                elif metric_name == 'Selected Features' and isinstance(metric_value, list):
                    scenario_metrics[metric_name] = ', '.join(metric_value)
                elif metric_name == 'Optimized Parameters' and isinstance(metric_value, dict):
                    for param_name, param_value in metric_value.items():
                        scenario_metrics[f'Opt_{param_name}'] = param_value

            metrics_data.append(scenario_metrics)

        return pd.DataFrame(metrics_data)