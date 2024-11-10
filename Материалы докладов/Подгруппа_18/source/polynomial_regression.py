from sklearn.model_selection import train_test_split
from overfitting_test import overfitting_mse
from sklearn.metrics import mean_squared_error
"""
Polynomial regression is a type of regression analysis that models the relationship
between a predictor x and the response y as an mth-degree polynomial:

y = β₀ + β₁x + β₂x² + ... + βₘxᵐ + ε

By treating x, x², ..., xᵐ as distinct variables, we see that polynomial regression is a
special case of multiple linear regression. Therefore, we can use ordinary least squares
(OLS) estimation to estimate the vector of model parameters β = (β₀, β₁, β₂, ..., βₘ)
for polynomial regression:

β = (XᵀX)⁻¹Xᵀy = X⁺y

where X is the design matrix, y is the response vector, and X⁺ denotes the Moore-Penrose
pseudoinverse of X. In the case of polynomial regression, the design matrix is

    |1  x₁  x₁² ⋯ x₁ᵐ|
X = |1  x₂  x₂² ⋯ x₂ᵐ|
    |⋮  ⋮   ⋮   ⋱ ⋮  |
    |1  xₙ  xₙ² ⋯  xₙᵐ|

In OLS estimation, inverting XᵀX to compute X⁺ can be very numerically unstable. This
implementation sidesteps this need to invert XᵀX by computing X⁺ using singular value
decomposition (SVD):

β = VΣ⁺Uᵀy

where UΣVᵀ is an SVD of X.

References:
    - https://en.wikipedia.org/wiki/Polynomial_regression
    - https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    - https://en.wikipedia.org/wiki/Numerical_methods_for_linear_least_squares
    - https://en.wikipedia.org/wiki/Singular_value_decomposition
"""

import matplotlib.pyplot as plt
import numpy as np


class PolynomialRegression:
    __slots__ = "degree", "params"

    def __init__(self, degree: int) -> None:
        if degree < 0:
            raise ValueError("Polynomial degree must be non-negative")

        self.degree = degree
        self.params = None

    @staticmethod
    def _design_matrix(data: np.ndarray, degree: int) -> np.ndarray:
        rows, *remaining = data.shape
        if remaining:
            raise ValueError("Data must have dimensions N x 1")

        return np.vander(data, N=degree + 1, increasing=True)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        X = PolynomialRegression._design_matrix(x_train, self.degree)  # noqa: N806
        _, cols = X.shape
        print("self.degree = ", self.degree)
        print("np.linalg.matrix_rank(X) = ", np.linalg.matrix_rank(X))
        if np.linalg.matrix_rank(X) < cols:
            raise ArithmeticError(
                "Design matrix is not full rank, can't compute coefficients"
            )

        # np.linalg.pinv() computes the Moore-Penrose pseudoinverse using SVD
        self.params = np.linalg.pinv(X) @ y_train
        print("self.params", self.params)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise ArithmeticError("Predictor hasn't been fit yet")

        return PolynomialRegression._design_matrix(data, self.degree) @ self.params



def main() -> None:
    import seaborn as sns

    mpg_data = sns.load_dataset("mpg")

    print("mpg_data", mpg_data)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(np.sort(mpg_data.weight), mpg_data.mpg.values,
                                                        test_size=0.3, random_state=0)

    poly_reg = PolynomialRegression(degree=3)
    poly_reg.fit(mpg_data.weight, mpg_data.mpg)

    weight_sorted = np.sort(mpg_data.weight)
    predictions = poly_reg.predict(weight_sorted)

    plt.scatter(mpg_data.weight, mpg_data.mpg, color="gray", alpha=0.5)
    plt.plot(weight_sorted, predictions, color="red", linewidth=3)
    plt.title("Predicting Fuel Efficiency Using Polynomial Regression")
    plt.xlabel("Weight (lbs)")
    plt.ylabel("Fuel Efficiency (mpg)")
    plt.show()

    overfitting_mse(x_train, x_test, y_train, y_test)

    # degrees = [1, 2, 3]
    # train_errors = []
    # test_errors = []
    #
    # for degree in degrees:
    #     poly_reg = PolynomialRegression(degree=degree)
    #
    #     poly_reg.fit(x_train, y_train)
    #
    #     predictions_train = poly_reg.predict(x_train)
    #     predictions_test = poly_reg.predict(x_test)
    #
    #     train_errors.append(mean_squared_error(y_train, predictions_train))
    #     test_errors.append(mean_squared_error(y_test, predictions_test))
    #
    # # Plot learning curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(degrees, train_errors, label='Train Error', marker='o')
    # plt.plot(degrees, test_errors, label='Test Error', marker='o')
    # plt.title('Learning Curves')
    # plt.xlabel('Polynomial Degree')
    # plt.ylabel('Mean Squared Error')
    # plt.xticks(degrees)
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # main()
