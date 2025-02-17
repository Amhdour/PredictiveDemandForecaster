import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processor import DataProcessor
from utils.forecasting import Forecaster
from utils.visualizations import Visualizer

def main():
    st.set_page_config(
        page_title="Demand Forecasting App",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Demand Forecasting Application")
    st.markdown("""
    This application helps you analyze and forecast demand patterns using various statistical methods.
    Upload your time series data and explore different forecasting models.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Data Processing
            data_processor = DataProcessor(uploaded_file)
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
            
            # Forecasting
            st.subheader("Forecasting")
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Select Forecasting Model", 
                                        ["Moving Average", "Exponential Smoothing", "SARIMA"])
                forecast_periods = st.number_input("Forecast Periods", min_value=1, value=30)
            
            with col2:
                if model_type == "Moving Average":
                    window_size = st.slider("Window Size", min_value=1, max_value=30, value=7)
                elif model_type == "Exponential Smoothing":
                    alpha = st.slider("Smoothing Factor (α)", min_value=0.0, max_value=1.0, value=0.3)
                elif model_type == "SARIMA":
                    seasonal_period = st.number_input("Seasonal Period", min_value=1, value=12)
            
            # Generate forecast
            forecaster = Forecaster()
            forecast_df, metrics = forecaster.generate_forecast(
                processed_df, 
                date_column, 
                target_column, 
                model_type,
                forecast_periods,
                window_size if model_type == "Moving Average" else None,
                alpha if model_type == "Exponential Smoothing" else None,
                seasonal_period if model_type == "SARIMA" else None
            )
            
            # Display forecast results
            st.subheader("Forecast Results")
            fig = visualizer.plot_forecast(processed_df, forecast_df, date_column, target_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame(metrics, index=[0])
            st.dataframe(metrics_df)
            
            # Download results
            st.download_button(
                label="Download Forecast Results",
                data=forecast_df.to_csv(index=False),
                file_name="forecast_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again.")

if __name__ == "__main__":
    main()
