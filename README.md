# Weather Forecasting-TimeSeries

## Overview
This project implements various time series forecasting models to predict daily temperature in Delhi based on historical climate data. The models include traditional statistical approaches (ARIMA, SARIMA) as well as deep learning methods (RNN, LSTM, Bidirectional LSTM).

## Dataset
The project uses the "Daily Delhi Climate" dataset which includes the following features:
- Mean temperature
- Humidity
- Wind speed
- Mean pressure

The data is split into training and testing sets with dates as the index.

## Project Structure
1. **Data Preprocessing**
   - Outlier detection and handling for mean pressure
   - Time series resampling (weekly, monthly)
   - Feature engineering (day, month, year, day of week)
   - Scaling (MinMax scaling for temperature, humidity, pressure; Robust scaling for wind speed)

2. **Exploratory Data Analysis**
   - Time series decomposition (trend, seasonality, residuals)
   - Seasonal patterns visualization
   - Correlation analysis
   - Moving average analysis

3. **Time Series Analysis**
   - Stationarity tests (ADF and KPSS)
   - ACF and PACF plots
   - Differencing to achieve stationarity

4. **Forecasting Models**
   - ARIMA (univariate)
   - SARIMA/SARIMAX (with exogenous variables)
   - Prophet (with and without additional regressors)
   - Simple RNN
   - LSTM
   - Bidirectional LSTM (multivariate)

5. **Model Evaluation**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² score
   - Visualization of predictions vs. actual values

## Key Findings
- The dataset shows clear yearly seasonality in all features
- Temperature shows an upward trend, potentially indicating climate change effects
- Humidity shows a decreasing trend, suggesting drier conditions
- 2016 was the hottest year in the dataset
- Deep learning models (especially Bidirectional LSTM) outperformed traditional statistical models
- Adding exogenous variables (humidity, wind speed, pressure) improved forecast accuracy

## Model Performance
| Model | R² Score |
|-------|----------|
| SARIMA | 0.28 |
| Prophet | 0.91 |
| Simple RNN | 0.92 |
| LSTM | 0.92 |
| Bidirectional LSTM | 0.93 |

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- tensorflow
- prophet

## How to Run
1. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow prophet
   ```

2. Load the dataset:
   ```python
   df_train = pd.read_csv('DailyDelhiClimateTrain.csv', index_col='date', parse_dates=True)
   ```

3. Run the notebook cells sequentially to reproduce the analysis and models.

## Future Improvements
- Hyperparameter tuning for deep learning models
- Exploring ensemble methods
- Incorporating additional features like precipitation
- Deployment as a web application for real-time forecasting 
