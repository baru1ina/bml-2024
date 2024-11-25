import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def classify(x, y):
    # Разбиваем датасет на тренировочный и тестовый
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values)

    # Берем модель
    nb_clf = GaussianNB()

    # Тренируем модель
    nb_clf.fit(x_train, y_train)

    # Предсказываем классификацию
    nb_clf_pred_res = nb_clf.predict(x_test)

    # Считаем accuracy score
    nb_clf_accuracy = accuracy_score(y_test, nb_clf_pred_res)
    print(nb_clf_accuracy)

    # Строим матрицу визуализации
    cm = confusion_matrix(y_test, nb_clf_pred_res)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")
    plt.show()


# Загружаем датасет iris
x1, y1 = load_iris(return_X_y=True, as_frame=True)
classify(x1, y1)

# Загружаем датасет breast_cancer
x2, y2 = load_breast_cancer(return_X_y=True, as_frame=True)
classify(x2, y2)

