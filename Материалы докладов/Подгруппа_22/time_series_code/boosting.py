import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from eda import prepare_data


def boosting(y_ds, test_size, minimal_pacf=0.1):
    X, y_ds = prepare_data(y_ds, test_size, minimal_pacf=minimal_pacf)

    X_train, X_test, y_train, y_test = train_test_split(X, y_ds, test_size=test_size, shuffle=False)

    xgb_model = XGBRegressor(n_estimators=50).fit(X_train, y_train)

    boost_predictions_train = pd.Series(
        xgb_model.predict(X_train),
        index=X_train.index,
        name='Prediction'
    )

    boost_predictions_test = pd.Series(
        xgb_model.predict(X_test),
        index=X_test.index,
        name='Prediction'
    )

    return boost_predictions_train, boost_predictions_test

