import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from source.NN_environment import fit_models, display_curves


# Модель
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0, l1_reg=0.0, l2_reg=0.0):
        super(LogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # линейный слой
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)  # слой dropout

        # Параметры регуляризации
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)  # функция активации - relu
        x = self.fc2(x)
        x = nn.functional.relu(x)  # функция активации - relu
        x = self.dropout(x)
        x = self.fc3(x)
        output = nn.functional.sigmoid(x)  # функция активации - sigmoid
        return output

    # L-регуляризация
    def regularize_loss(self):  #
        l1_loss = 0
        l2_loss = 0

        if self.l1_reg == 0 and self.l2_reg == 0:
            return 0.0

        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param * param)
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss


def main():
    # Загрузка датасета Титаник
    data = pd.read_csv("data/titanic.csv")

    # Предварительная обработка данных
    data = data.drop(["PassengerId", "Lname", "Name", "Ticket", "Cabin"], axis=1)  # удаляем нерелевантные столбцы
    data.fillna({"Age": data["Age"].median()}, inplace=True)  # заполняем пропущенные значения в столбце "Age" медианным значением
    data = pd.get_dummies(data, columns=["Sex", "Embarked"])  # преобразуем категориальные признаки в числовые с помощью One-Hot Encoding

    # Разделение на обучающие и тестовые данные
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Преобразование в тензоры PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

    # Создаем DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Определение размера входных данных
    input_size = X_train_tensor.shape[1]

    # Создание моделей
    model = LogisticRegressionModel(input_size)
    model_dropout = LogisticRegressionModel(input_size, dropout_rate=0.15)
    model_l1 = LogisticRegressionModel(input_size, l1_reg=0.005)
    model_l2 = LogisticRegressionModel(input_size, l2_reg=0.01)

    # Определение функции потерь и оптимизаторов для каждой модели
    criterion = nn.BCELoss()  # Binary Cross Entropy loss
    l_r = 0.001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=l_r)
    optimizer_l1 = torch.optim.Adam(model_l1.parameters(), lr=l_r)
    optimizer_l2 = torch.optim.Adam(model_l2.parameters(), lr=l_r)

    # Количества эпох
    epochs = 200

    # Обучение моделей
    histories = fit_models(epochs, [model, model_dropout, model_l1, model_l2],
                           [optimizer, optimizer_dropout, optimizer_l1, optimizer_l2], train_loader, test_loader, criterion, t_type="reg")

    # Построение и вывод графиков кривых обучения
    display_curves(histories)


if __name__ == '__main__':
    main()
