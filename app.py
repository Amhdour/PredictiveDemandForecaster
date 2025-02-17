import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processor import DataProcessor
from utils.forecasting import Forecaster
from utils.visualizations import Visualizer
from utils.scenario_comparison import ScenarioManager

def main():
    st.set_page_config(
        page_title="Demand Forecasting App",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Demand Forecasting Application")
    st.markdown("""
    This application helps you analyze and forecast demand patterns using various statistical methods.
    Upload your time series data and explore different forecasting models, including advanced options like ARIMA and Prophet.
    """)

    # Initialize session state for scenario management
    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()

    # Data Loading Options
    data_option = st.radio(
        "Choose data source",
        ["Upload your own data", "Use sample data"],
        horizontal=True
    )

    uploaded_file = None
    if data_option == "Upload your own data":
        uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=['csv', 'xlsx'])

    # Initialize data processor
    data_processor = DataProcessor(uploaded_file)

    # Load data based on selection
    if data_option == "Use sample data" or uploaded_file is not None:
        try:
            df = data_processor.load_data()

            # Data Preview
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Data Preprocessing Options
            st.subheader("Data Preprocessing")
            col1, col2 = st.columns(2)

            with col1:
                date_column = st.selectbox("Select Date Column", df.columns)
                target_column = st.selectbox("Select Target Column (Values to Forecast)", 
                                            df.select_dtypes(include=['float64', 'int64']).columns)

            with col2:
                handle_missing = st.selectbox("Handle Missing Values", 
                                           ["Forward Fill", "Backward Fill", "Linear Interpolation"])

            # Process data based on selections
            processed_df = data_processor.preprocess_data(df, date_column, target_column, handle_missing)

            # Visualization
            st.subheader("Time Series Visualization")
            visualizer = Visualizer()
            fig = visualizer.plot_time_series(processed_df, date_column, target_column)
            st.plotly_chart(fig, use_container_width=True)

            # Feature Selection and Parameter Optimization Options
            st.subheader("Advanced Options")
            col1, col2 = st.columns(2)

            with col1:
                use_feature_selection = st.checkbox("Enable Automated Feature Selection", value=True)

            with col2:
                optimize_parameters = st.checkbox("Enable Parameter Optimization", value=False,
                                               help="Automatically find the best parameters for the selected model")

            # Scenario Management section
            st.subheader("Scenario Management")

            # Scenario creation
            with st.expander("Create New Forecast Scenario", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    scenario_name = st.text_input("Scenario Name", 
                                                value=f"Scenario {len(st.session_state.scenario_manager.get_scenario_names()) + 1}")
                    model_type = st.selectbox("Select Forecasting Model", 
                                          ["Moving Average", "Exponential Smoothing", "ARIMA", "Prophet"],
                                          key="model_type")
                    forecast_periods = st.number_input("Forecast Periods", min_value=1, value=30,
                                                     key="forecast_periods")

                with col2:
                    if not optimize_parameters:
                        # Model-specific parameters (manual configuration)
                        if model_type == "Moving Average":
                            window_size = st.slider("Window Size", min_value=1, max_value=30, value=7,
                                                  key="window_size")
                            params = {"window_size": window_size}
                        elif model_type == "Exponential Smoothing":
                            alpha = st.slider("Smoothing Factor (α)", min_value=0.0, max_value=1.0, value=0.3,
                                            key="alpha")
                            params = {"alpha": alpha}
                        elif model_type == "ARIMA":
                            p = st.slider("AR Order (p)", min_value=0, max_value=5, value=1, key="p")
                            d = st.slider("Differencing Order (d)", min_value=0, max_value=2, value=1, key="d")
                            q = st.slider("MA Order (q)", min_value=0, max_value=5, value=1, key="q")
                            params = {"p": p, "d": d, "q": q}
                        elif model_type == "Prophet":
                            st.info("Prophet uses automatic parameter optimization")
                            params = {}
                    else:
                        st.info(f"Parameters will be optimized automatically for {model_type} model")
                        params = {}

                # Generate forecast for scenario
                if st.button("Add Scenario"):
                    with st.spinner(f"Generating forecast for {scenario_name}..." + 
                                 (" (Optimizing parameters)" if optimize_parameters else "")):
                        forecaster = Forecaster()
                        forecast_df, metrics = forecaster.generate_forecast(
                            processed_df, 
                            date_column, 
                            target_column, 
                            model_type,
                            forecast_periods,
                            use_feature_selection=use_feature_selection,
                            optimize_parameters=optimize_parameters,
                            **params
                        )

                        # Add scenario to manager
                        st.session_state.scenario_manager.add_scenario(
                            scenario_name,
                            model_type,
                            metrics.get('Optimized Parameters', params),
                            forecast_df,
                            metrics
                        )
                        st.success(f"Added scenario: {scenario_name}")

                        # Display optimization results if used
                        if optimize_parameters and 'Optimized Parameters' in metrics:
                            st.info(f"Optimized parameters: {metrics['Optimized Parameters']}")

            # Scenario Comparison
            if st.session_state.scenario_manager.get_scenario_names():
                st.subheader("Compare Scenarios")

                # Select scenarios to compare
                selected_scenarios = st.multiselect(
                    "Select scenarios to compare",
                    st.session_state.scenario_manager.get_scenario_names(),
                    default=st.session_state.scenario_manager.get_scenario_names()
                )

                if selected_scenarios:
                    # Plot comparison
                    fig = st.session_state.scenario_manager.plot_comparison(
                        processed_df,
                        date_column,
                        target_column,
                        selected_scenarios
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display metrics comparison
                    st.subheader("Metrics Comparison")
                    metrics_df = st.session_state.scenario_manager.compare_metrics()
                    st.dataframe(metrics_df)

                    # Download results
                    st.download_button(
                        label="Download Comparison Results",
                        data=metrics_df.to_csv(index=False),
                        file_name="forecast_comparison.csv",
                        mime="text/csv"
                    )

                # Option to clear scenarios
                if st.button("Clear All Scenarios"):
                    st.session_state.scenario_manager.clear_scenarios()
                    st.success("All scenarios cleared")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again.")

if __name__ == "__main__":
    main()