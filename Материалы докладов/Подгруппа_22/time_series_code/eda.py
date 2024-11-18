import pandas as pd
from scipy.signal import periodogram
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


def draw_periodogram(data_list):
    frequencies, powers = periodogram(data_list, fs=1)
    plt.plot(frequencies, powers)
    plt.title('')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()


def draw_PACF(series, lags=20):
    # PACF plot
    plot_pacf(series, lags=lags)
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation Function (PACF) Plot')
    plt.grid(True)
    plt.show()


def make_lags(df, n_lags=1, lead_time=1):
    """
    Compute lags of a pandas.Series from lead_time to lead_time + n_lags. Alternatively, a list can be passed as n_lags.
    Returns a pd.DataFrame whose ith column is either the i+lead_time lag or the ith element of n_lags.
    """
    if isinstance(n_lags, int):
        lag_list = list(range(lead_time, n_lags + lead_time))
    else:
        lag_list = n_lags

    lags = {
        f'{df.name}_lag_{i}': df.shift(i) for i in lag_list
    }

    return pd.concat(lags, axis=1)


def prepare_data(series, test_size,
                 to_predict=1,
                 nlags=20,
                 minimal_pacf=0.1):
    '''
    Creates a feature dataframe by making lags and a target series by a negative to_predict-shift.
    Returns X, y.
    '''
    s = series.iloc[:-test_size]

    if isinstance(to_predict, int):
        to_predict = [to_predict]

    draw_PACF(s, lags=nlags)

    s_pacf = pd.Series(pacf(s, nlags=nlags))
    column_list = s_pacf[abs(s_pacf) > minimal_pacf].index

    X = make_lags(series, n_lags=column_list).dropna()
    y = make_lags(series, n_lags=[-x for x in to_predict]).loc[X.index].squeeze()
    return X, y
