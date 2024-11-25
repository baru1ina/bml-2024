import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def train(model, train_loader, optimizer, criterion, epoch, t_type, verbose=False):
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)  # прогноз модели
        if t_type == "reg":
            output = output.squeeze()

        loss = criterion(output,
                         target) + model.regularize_loss()  # ошибка на данном этапе обучения (с L-регуляризацией)
        train_loss += criterion(output,
                                target).item()  # добавляем в общую ошибку на данном этапе обучения (без L-регуляризации)

        if t_type == "class":
            pred = output.argmax(dim=1, keepdim=True)  # определение класса по максимальному значению вероятности
        else:
            pred = (output > 0.75).float()
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()  # обратное распределение
        optimizer.step()

        if batch_idx % 10 == 0 and verbose:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def evaluate(model, test_loader, criterion, t_type, verbose=False):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  # прогноз модели
            if t_type == "reg":
                output = output.squeeze()

            test_loss += criterion(output, target).item()  # просуммированная loss патча

            if t_type == "reg":
                pred = (output > 0.75).float()
            else:
                pred = output.argmax(dim=1, keepdim=True)  # определение класса по максимальному значению вероятности
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

    return test_loss, correct / len(test_loader.dataset)


def display_curves(histories, names=None):
    n = histories.shape[0]
    if names is None:
        names = ['Base', 'Dropout', 'L1-regularization', 'L2-regularization']
    fig, axs = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.1}, sharex=True, sharey=True)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.suptitle("Кривые обучения для различных методов регуляризации", fontweight='bold', fontsize='x-large')

    fig2, axs2 = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.1}, sharex=True, sharey=True)
    fig2.set_figwidth(10)
    fig2.set_figheight(8)
    fig2.suptitle("Кривые точности (acc) для различных методов регуляризации", fontweight='bold', fontsize='x-large')

    for i in range(n):
        axs[i // 2, i % 2].plot(histories[i, 0, :], label=f'Train')
        axs[i // 2, i % 2].plot(histories[i, 2, :], label=f'Test')
        if i // 2 != 0:
            axs[i // 2, i % 2].set_xlabel('Epoch')
        if i % 2 == 0:
            axs[i // 2, i % 2].set_ylabel('Loss')
        axs[i // 2, i % 2].set_title(names[i], fontsize='medium')

        axs[i // 2, i % 2].grid(which='major', color='#DDDDDD', linewidth=0.9)
        axs[i // 2, i % 2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axs[i // 2, i % 2].minorticks_on()
        axs[i // 2, i % 2].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i // 2, i % 2].legend(loc="upper left")

        axs2[i // 2, i % 2].plot(histories[i, 1, :], label=f'Train')
        axs2[i // 2, i % 2].plot(histories[i, 3, :], label=f'Test')
        if i // 2 != 0:
            axs2[i // 2, i % 2].set_xlabel('Epoch')
        if i % 2 == 0:
            axs2[i // 2, i % 2].set_ylabel('Acc')
        axs2[i // 2, i % 2].set_title(names[i], fontsize='medium')

        axs2[i // 2, i % 2].grid(which='major', color='#DDDDDD', linewidth=0.9)
        axs2[i // 2, i % 2].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axs2[i // 2, i % 2].minorticks_on()
        axs2[i // 2, i % 2].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs2[i // 2, i % 2].legend(loc="lower right")

    plt.show()


def fit_models(epochs, models, optimizers, train_loader, test_loader, criterion, t_type="class"):
    # Объявление массивов значений потерь и точности (характеристики)
    histories = [[] for i in range(len(models))]  # [model: [epoch: ["loss": float, "acc": float, "val_loss": float, "val_acc": float]]]
    # Обучение и тестирование модели
    for epoch in range(1, epochs + 1):
        print(f"- Epoch {epoch}/{epochs} - |", end="")
        start_tic = time.time()
        for m_i, model in enumerate(models):
            loss, acc = train(model, train_loader, optimizers[m_i], criterion, epoch, t_type)  # обучение модели на i эпохе
            print(".", end="")
            val_loss, val_acc = evaluate(model, test_loader, criterion, t_type)  # тестирование модели на i эпохе
            histories[m_i].append([loss, acc, val_loss, val_acc])  # сохранение значений характеристик
            print(".", end="")

        cur_time = time.time() - start_tic
        print(f"| - Took {int(cur_time // 60)}:{cur_time % 60 :.3f} s")

    # Для удобства переводим массивы в numpy.array и транспонируем (получаем массив зависимостей характеристик от эпохи)
    return np.transpose(np.array(histories), (0, 2, 1))
