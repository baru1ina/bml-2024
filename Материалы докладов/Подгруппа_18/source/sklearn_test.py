from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Define the input and output data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1.2, 2.2, 2.8, 4.0, 5.1])

# Transform the input data into polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
reg = LinearRegression().fit(X_poly, y)

# Predict the output for a new input
new_X = np.array([6]).reshape(-1, 1)

# new_X = np.array([6, 7, 8]).reshape(-1, 1)

new_X_poly = poly.transform(new_X)


def print_res():
    print("X: ", X)
    print("y: ", y)
    print("X_poly: ", X_poly)
    print("reg.coef_: ", reg.coef_)
    print("new_X: ", new_X)
    print("new_X_poly: ", new_X_poly)
    print(reg.predict(new_X_poly))


if __name__ == "__main__":
    print_res()
