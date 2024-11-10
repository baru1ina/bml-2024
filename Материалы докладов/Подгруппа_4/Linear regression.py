import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.array([[10], [20], [30], [40], [50]])
y = np.array([20, 40, 60, 80, 100])

# Добавляем случайный шум (нормальное распределение с нулевым средним и дисперсией 2)
epsilon = np.random.normal(0, 2, size=y.shape)
y_noisy = y + epsilon

# Добавляем столбец единиц для свободного члена
x_b = np.c_[np.ones((x.shape[0], 1)), x]

# Находим параметры регрессии
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_noisy)

print("Свободный член:", theta_best[0])
print("Коэффициент (наклон):", theta_best[1])

# Функция для предсказаний
def predict(x, theta):
    x_b = np.c_[np.ones((x.shape[0], 1)), x]
    return x_b.dot(theta)

# Прогноз для нового значения
new_value = np.array([[60]])
predicted_value = predict(new_value, theta_best)
print("Прогнозируемый объем продаж для рекламного бюджета:", predicted_value[0])

# Построение графика
plt.figure(figsize=(8, 6))

# Линия регрессии
x_line = np.linspace(0, 60, 100).reshape(-1, 1)
y_line = predict(x_line, theta_best)
plt.plot(x_line, y_line, "b-", label="Линия регрессии")

# Исходные данные с шумом
plt.scatter(x, y_noisy, color="red", label="Исходные данные")

# Точка для нового значения
plt.scatter(new_value, predicted_value, color="green", label="Прогноз")

# Настройки графика
plt.xlabel("Рекламный бюджет")
plt.ylabel("Объем продаж")
plt.legend()
plt.title("Линейная регрессия для прогнозирования объема продаж")
plt.grid(True)

plt.show()
