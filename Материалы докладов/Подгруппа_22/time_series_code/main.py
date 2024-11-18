import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.metrics import r2_score
from linear_model import trend_and_season
from boosting import boosting


def get_hybrid_prediction(series: pd.Series,
                          test_size,
                          season_freq='YE',
                          fourier_order=0,
                          constant=True,
                          dp_order=1,
                          dp_drop=True,
                          fourier=None,
                          to_predict=1,
                          nlags=20,
                          minimal_pacf=0.1):

    data_series = series[:-test_size]

    y_ds, y_s = trend_and_season(data_series,
                                 season_freq=season_freq,
                                 fourier_order=fourier_order,
                                 dp_order=dp_order,
                                 dp_drop=dp_drop,
                                 fourier=fourier,
                                 constant=constant,
                                 size_prediction=test_size)

    boost_pred_train, boost_pred_test = boosting(y_ds, test_size, minimal_pacf=minimal_pacf)

    predictions_train = boost_pred_train + y_s[boost_pred_train.index]

    predictions_test = boost_pred_test + y_s[boost_pred_test.index]

    print(f'R2 train score: {r2_score(series.loc[predictions_train.index][:-to_predict], predictions_train[:-to_predict])}')

    print(f'R2 test score: {r2_score(series.loc[predictions_test.index][:-to_predict], predictions_test[:-to_predict])}')

    draw_graphics(series, predictions_train, predictions_test)


def draw_graphics(series, prediction_train, prediction_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    y_train_ps = series.loc[prediction_train.index]
    y_test_ps = series.loc[prediction_test.index]
    y_train_ps.plot(ax=ax1, legend=True)
    prediction_train.plot(ax=ax1, label="prediction train", legend=True)
    ax1.set_title('Train Predictions')

    y_test_ps.plot(ax=ax2, legend=True)
    prediction_test.plot(ax=ax2, label="prediction test", legend=True)
    ax2.set_title('Test Predictions')
    plt.show()

    plt.figure()
    index = prediction_train.index.union(prediction_test.index)
    plot_series = series.loc[index]
    plot_series.plot(color='blue', legend=True)
    prediction_train.plot(color='orange', label="prediction train", legend=True)
    prediction_test.plot(color='red', label="prediction test", legend=True)
    plt.show()


if __name__ == "__main__":
    data = yf.download('BTC-USD',
                       start='2020-02-14',
                       end='2022-09-21',
                       interval='1d'
                       )

    data_series = data['Close']['BTC-USD']
    data_series.index = pd.DatetimeIndex(data_series.index.date)
    get_hybrid_prediction(data_series, 60,
                          season_freq='YE',
                          fourier_order=4,
                          constant=True,
                          dp_order=5,
                          dp_drop=True,
                          fourier=None,
                          to_predict=1,
                          nlags=20,
                          minimal_pacf=0.05)
