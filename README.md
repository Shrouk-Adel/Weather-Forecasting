# Weather Forecasting Project

A comprehensive time series analysis and forecasting project using Delhi climate data to predict mean temperature.

## Overview

This project explores various time series forecasting techniques to predict daily weather patterns in Delhi. The analysis includes data cleaning, exploratory data analysis, feature engineering, and implementation of multiple forecasting models ranging from traditional statistical approaches to advanced deep learning architectures.

## Dataset

The project uses the "Daily Delhi Climate" dataset which includes the following features:
- Mean temperature
- Humidity
- Wind speed
- Mean pressure

The data is split into training and testing sets to evaluate model performance.

## Methodology

### Data Preprocessing
- Outlier detection and handling in mean pressure data
- Feature engineering including day, month, year, and day of week
- Time series decomposition to identify trend, seasonality, and residuals
- Stationarity testing using ADF and KPSS tests
- Data scaling using MinMax and Robust scaling techniques

### Models Implemented

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Basic implementation for baseline comparison

2. **SARIMA (Seasonal ARIMA)**
   - Enhanced with seasonal components to capture weekly patterns
   - Incorporated exogenous variables (humidity, wind speed, mean pressure)

3. **SARIMAX (Seasonal ARIMA with Exogenous Variables)**
   - Advanced version using differenced target values
   - Leverages multiple weather features for improved predictions

4. **Prophet**
   - Facebook's forecasting tool with excellent handling of seasonality
   - Implemented both with and without additional regressor variables

5. **SimpleRNN (Recurrent Neural Network)**
   - Univariate implementation with a window size of 5
   - Early stopping to prevent overfitting

6. **LSTM (Long Short-Term Memory)**
   - Advanced deep learning approach for sequence forecasting
   - Multi-layer architecture with ReLU and linear activation functions

7. **Bidirectional LSTM**
   - Multivariate implementation using all weather features
   - Captures information from both past and future time steps

## Results

Among all models tested, **Prophet** demonstrated the best balance of performance, simplicity, and computational efficiency:

- Achieved an R² score of approximately 91% with all features
- Required minimal parameter tuning compared to complex deep learning models
- Demonstrated robustness to outliers and irregular data patterns
- Provided useful decomposition of the forecast into trend and seasonal components

While deep learning models showed competitive performance, Prophet delivered comparable results with significantly less complexity and computational requirements.

## Key Insights

- All weather features show strong yearly seasonality
- Temperature in Delhi shows a possible warming trend after 2015
- Humidity displays a slight decreasing trend over time
- Wind speed exhibits highly variable seasonal patterns
- Mean pressure fluctuations indicate changes in weather systems

## Technologies Used

- Python
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for visualization
- Statsmodels for statistical modeling
- Prophet for advanced forecasting
- TensorFlow/Keras for deep learning models
- Scikit-learn for preprocessing and evaluation metrics

## Getting Started

1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn statsmodels prophet tensorflow scikit-learn
   ```

2. Load and preprocess the data:
   ```python
   df_train = pd.read_csv('DailyDelhiClimateTrain.csv', index_col='date', parse_dates=True)
   df_test = pd.read_csv('DailyDelhiClimateTest.csv', index_col='date', parse_dates=True)
   ```

3. Run the models as shown in the notebook.

## Future Work

- Incorporate additional meteorological features
- Experiment with ensemble methods combining multiple forecasting techniques
- Extend the analysis to multiple geographical locations
- Implement real-time prediction capabilities with API integration
- Explore the impact of climate change on weather patterns using longer time series

## Conclusion

Time series forecasting for weather data presents unique challenges due to complex seasonal patterns and non-linear relationships. This project demonstrates that while advanced deep learning approaches can achieve high accuracy, simpler models like Prophet often provide an excellent balance between performance, interpretability, and computational efficiency.
