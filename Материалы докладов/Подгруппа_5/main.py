import numpy as np
from matplotlib import cm
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from Logit_reg import logistic_regression

def visualization():
    (X, y) = make_blobs(n_samples=1500, centers=2, n_features=2,
                        random_state=20)

    log_reg = logistic_regression(X.shape[1])
    log_reg.fit(X, y, alpha=0.01, epochs=100)

    (W0, W1, W2) = log_reg.get_weights()

    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 1500)
    D_B = (-W0 - (W1 * x_vals)) / W2

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.jet)
    plt.plot(x_vals, D_B, "r-")
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()


def my_model(X_train, X_test, Y_train, Y_test):

    # Model training
    model = logistic_regression(X.shape[1])

    model.fit(X_train, Y_train, alpha=0.001, epochs=100)

    # Prediction on test set
    Y_pred = model.predict(X_test)
    print(classification_report(Y_pred, Y_test))

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # roc curve for tpr = fpr
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def SL_model(X_train, X_test, Y_train, Y_test):
    model1 = LogisticRegression(solver='liblinear', random_state=0)
    param_grid = {
        'C': [0.00001,0.001, 0.01, 0.1, 1, 10],
        'solver': ['liblinear','newton-cg'],
        'max_iter': [100, 1000,10000]
    }
    clf = GridSearchCV(model1, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X_train, Y_train)
    best_clf.best_estimator_
    Y_pred = best_clf.predict(X_test)

    print(classification_report(Y_test, Y_pred))

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    roc_auc
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # roc curve for tpr = fpr
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("diabetes.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values
    Y = Y.reshape(-1)

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=0)

    #my_model(X_train, X_test, Y_train, Y_test) # для использования модели из Logit_reg.py
    SL_model(X_train, X_test, Y_train, Y_test) # для использования модели из sklearn