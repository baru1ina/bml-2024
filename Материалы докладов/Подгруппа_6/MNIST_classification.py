import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from source.NN_environment import fit_models, display_curves


class Net(nn.Module):
    def __init__(self, dropout_rate=0.0, l1_reg=0.0, l2_reg=0.0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # свёрточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # свёрточный слой
        self.pool = nn.MaxPool2d(2, 2)  # уменьшение размерности карты признаков
        self.dropout = nn.Dropout(dropout_rate)  # слой dropout
        self.fc1 = nn.Linear(9216, 128)  # линейный слой
        self.fc2 = nn.Linear(128, 10)  # линейный слой

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    # Прямой проход нейронной сети
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)  # функция активации - relu
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # распрямляющий слой
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)  # выходные значения определяются с помощью ф-ции активации log_softmax
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
    # Загрузка датасета MNIST (обучающая и тестовые выборки загружаются отдельно)
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train_indices = torch.randperm(len(train_dataset))[:5000]  # рандомно выбираем 10000 элементов для тренировочной выборки

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_indices = torch.randperm(len(test_dataset))[:1000]  # рандомно выбираем 5000 элементов для тестовой выборки

    # Создаем DataLoader
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=64, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=100, shuffle=False)

    # Создание моделей
    model = Net()
    model_dropout = Net(dropout_rate=0.2)
    model_l1 = Net(l1_reg=0.001)
    model_l2 = Net(l2_reg=0.01)

    # Определение функции потерь и оптимизаторов для каждой модели
    criterion = nn.CrossEntropyLoss()

    l_r = 0.001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=l_r)
    optimizer_l1 = torch.optim.Adam(model_l1.parameters(), lr=l_r)
    optimizer_l2 = torch.optim.Adam(model_l2.parameters(), lr=l_r)

    # Определение количества эпох
    epochs = 50

    # Обучение моделей
    histories = fit_models(epochs, [model, model_dropout, model_l1, model_l2], [optimizer, optimizer_dropout, optimizer_l1, optimizer_l2], train_loader, test_loader, criterion, t_type="class")

    # Построение и вывод графиков кривых обучения
    display_curves(histories)


if __name__ == '__main__':
    main()
