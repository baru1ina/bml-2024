import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def overfitting_mse(x_train, x_test, y_train, y_test):
    degrees = [i for i in range(1, 7)]
    train_errors = []
    test_errors = []

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        x_poly_train = poly_features.fit_transform(x_train[:, np.newaxis])
        x_poly_test = poly_features.transform(x_test[:, np.newaxis])

        model = LinearRegression()
        model.fit(x_poly_train, y_train)

        train_predictions = model.predict(x_poly_train)
        test_predictions = model.predict(x_poly_test)

        train_errors.append(mean_squared_error(y_train, train_predictions))
        test_errors.append(mean_squared_error(y_test, test_predictions))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, label='Train Error', marker='o')
    plt.plot(degrees, test_errors, label='Test Error', marker='o')
    plt.title('Learning Curves')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()




