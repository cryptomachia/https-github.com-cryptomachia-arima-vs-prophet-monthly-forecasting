import time
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()
returns_list = [44.05, 61.77, 172.76, 50.01, -8.56, -29.89, 9.6, 30.42, 60.79, 449.35, -34.81,
                10.03, -31.03, -17.25, -1.6, 39.46, 2.2, -9.69, -17.55, -19.01, -12.95, 12.82, -15.11,
                -33.05, 18.43, -4.38, -3.46, -3.17, 15.19, 8.2, -18.67, 2.35, 33.49, 19.27, 13.83,
                -14.83, 20.08, -5.35, 7.27, 18.78, 27.14, -7.67, -7.49, 6.04, 14.71, 5.42, 30.8,
                -0.04, 23.07, -9.05, 32.71, 52.71, 10.45, 17.92, 65.32, -7.44, 47.81, 53.48, 38.89,
                -25.41, 0.47, -32.85, 33.43, -18.99, -14.62, 20.96, -9.27, -5.58, -3.83, -36.57, -5.15,
                -8.58, 11.14, 7.05, 34.36, 52.38, 26.67, -6.59, -4.6, -13.38, 10.17, -17.27, -5.15,
                29.95, -8.6, -24.92, 34.26, 9.51, -3.18, 24.03, 2.83, -7.51, 27.77, 42.95, 46.92,
                14.51, 36.78, 29.84, -1.98, -35.31, -5.95, 18.19, 13.8, -7.03, 39.93, -7.11, -18.9,
                -16.68, 12.21, 5.39, -17.3, -15.6, -37.28, 16.8, -13.88, -3.12, 5.56, -16.23, -3.59,
                39.63, 0.03, 22.96, 2.81, -6.98, 11.98, 0.11]

while len(returns_list) < 128:
    returns_list.append(np.nan)

data = {'Date': pd.date_range(start="2013-01", periods=128, freq='M'), 'returns': returns_list}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)
train = df['returns'].dropna()[:-12]
test = df['returns'].dropna()[-12:]

def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.5)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                print(f"Evaluating ARIMA{order} ...")
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg


p_values, d_values, q_values = [0, 1, 2, 3, 4, 5, 6, 7, 8], range(0, 4), range(0, 4)
best_order = evaluate_models(train.values, p_values, d_values, q_values)

predictions_arima, actual = [], []
for t in range(len(test)):
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    predictions_arima.append(output.iloc[0])
    obs = test.iloc[t]
    actual.append(obs)
    new_date = train.index[-1] + pd.DateOffset(months=1)
    train = train.append(pd.Series({new_date: obs}))

train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
train_df['ds'] = pd.to_datetime(train_df['ds'])

final_prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                              n_changepoints=50, changepoint_range=0.9, changepoint_prior_scale=0.2)
final_prophet_model.add_seasonality(name='half-yearly', period=6, fourier_order=6)
final_prophet_model.fit(train_df)
future_final = final_prophet_model.make_future_dataframe(periods=len(test), freq='M')
forecast_final = final_prophet_model.predict(future_final)
predictions_prophet = forecast_final['yhat'][-len(test)-1:-1].values

end_time = time.time()
runtime = (end_time - start_time) / 60
print(f"training runtime: {runtime:.2f} minutes")

plt.figure(figsize=(15, 6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.plot(pd.date_range(start="2022-10", periods=len(predictions_arima), freq='M'), predictions_arima, color='red', linestyle='-', marker='o', label="ARIMA Predictions")
plt.plot(pd.date_range(start="2022-10", periods=len(predictions_prophet), freq='M'), predictions_prophet, color='green', linestyle='-', marker='o', label="Prophet Predictions")
plt.plot(pd.date_range(start="2022-10", periods=len(actual), freq='M'), actual, color='blue', linestyle='-', marker='o', label="Actual")

plt.xlabel("Date")
plt.ylabel("Returns")
plt.title("Rolling Forecast vs Actual")
plt.legend(loc="best")
plt.tight_layout()

plt.show()
print(f"Predicted return for October 2023 using ARIMA: {predictions_arima[-1]:.2f}%")
print(f"Predicted return for October 2023 using Prophet: {predictions_prophet[-1]:.2f}%")

mse_arima = mean_squared_error(actual, predictions_arima)
mse_prophet = mean_squared_error(actual, predictions_prophet)
print(f"MSE for ARIMA: {mse_arima:.2f}")
print(f"MSE for Prophet: {mse_prophet:.2f}")