# Time Series Forecasting: ARIMA vs Facebook Prophet
Considering the strong postive correlation between altcoin and BTC prices, and altcoin prices movement lag behind BTC, this weekend project tries to predict the BTC returns for the next month, given prior historical training data in monthly returns. At time of writing, I forecasting for the October 2023 monthly percentage return.

## Required Packages
`pip install pandas numpy statsmodels matplotlib fbprophet scikit-learn`
I made minor changes to facebooks forecasting.py as Prophet module was outdated, and also resolved version conflicts by manually installing specific versions of modules. To make it easy, simply use my venv folder with all the changes by dragging venv folder into pythonProject, or wherever you want it.
`cd /Users/name/PycharmProjects/pythonProject`
`source venv_name/bin/activate`


### Setup and Preprocessing
- **Timestamp Initialization**: Utilizes the `time` library to record the current runtime, aiding in performance analysis.
- **Data Structuring**:
  - We start with a dataset named `returns_list`, representing historical BTC returns.
  - This dataset is molded into a pandas `DataFrame` (`df`) with monthly intervals, initiating from January 2013.

### ARIMA Implementation
- **Model Evaluation Utility**: The function `evaluate_arima_model`:
  1. Divides the dataset into training and test subsets.
  2. Deploys the ARIMA model with provided `p`, `d`, and `q` parameters for training.
  3. Forecasts against the test subset.
  4. Measures forecast accuracy via Mean Squared Error (MSE).
  
- **Parameter Selection**: Within the `evaluate_models` function:
  1. Experiments span across diverse ARIMA parameter combinations. Here:
     - `p_values` depict the autoregressive terms' order, spanning from 0 to 8.
     - `d_values` indicate the count of differencing needed for series stationarity, assessed from 0 to 3.
     - `q_values` outline the moving average order, also ranged from 0 to 3.
  2. The aim remains to pinpoint the parameter ensemble yielding the lowest MSE.

- **Forecasting with ARIMA**:
  1. Employing the optimal parameters, ARIMA predictions are churned in a rolling manner.
  2. Post every forecast, the model assimilates the latest test data point into its training subset.


### Prophet Implementation
- **Model Configuration**:
  1. The Prophet model stands configured to majorly discern yearly patterns (`yearly_seasonality=True`), sidelining daily and weekly patterns.
  2. The directive `n_changepoints=50` conveys that up to 50 potential trend shift points are considered by the model.
  3. `changepoint_range=0.9` implies that only the initial 90% of the dataset is taken into account for potential changepoints.
  4. `changepoint_prior_scale=0.2` serves as a regularizer, with larger values granting trend adaptability and smaller ones imposing rigidity.
  5. Custom seasonality emerges with a `half-yearly` frequency. The `period=6` represents a 6-month cyclic pattern, while `fourier_order=6` details the seasonality's complexity.

  - Graph compares ARIMA forecasts, Prophet projections, and the actual monthly datapoints.

