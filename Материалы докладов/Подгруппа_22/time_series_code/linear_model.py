import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import LinearRegression
import pandas as pd


def trend_and_season(series_: pd.Series,
                     season_freq='YE',
                     fourier_order=0,
                     dp_order=1,
                     dp_drop=True,
                     fourier=None,
                     constant=True,
                     size_prediction=60):

    """Передаём сразу обучающую часть в series_"""
    if fourier is None:
        fourier = CalendarFourier(freq=season_freq, order=fourier_order)

    dp = DeterministicProcess(
        index=series_.index,
        constant=constant,
        order=dp_order,
        additional_terms=[fourier],
        drop=dp_drop
    )

    X_in = dp.in_sample()

    index_ = pd.date_range(X_in.index[-1], periods=size_prediction+1).tolist()
    X_out = dp.out_of_sample(size_prediction, forecast_index=index_[1:])

    lin_model = LinearRegression().fit(X_in, series_)

    X = pd.concat([X_in, X_out], axis=0)

    y_s = pd.Series(
        lin_model.predict(X),
        index=X.index,
        name=series_.name + '_pred'
    )
    y_s.name = series_.name

    # y deseasonal
    y_ds = series_ - y_s
    y_ds.name = series_.name + '_deseasoned'
    return y_ds, y_s

