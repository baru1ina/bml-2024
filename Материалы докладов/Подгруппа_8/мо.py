import numpy as np


def rastrigin(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def rastrigin_grad(x, y, A=10):
    dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
    return np.array([dx, dy])


# Общая функция оптимизации с проверкой на сходимость
def optimize_convergence(method, grad_func, lr, init_point, tol=1e-6, max_steps=10000, log_trajectory=False, **kwargs):
    point = np.array(init_point, dtype=np.float64)
    state = kwargs.get("state", {})
    trajectory = [point] if log_trajectory else None

    for step in range(1, max_steps + 1):
        grad = grad_func(*point)
        prev_point = point
        point, state = method(point, grad, lr, state, **kwargs)

        if log_trajectory:
            trajectory.append(point)

        # Проверка на сходимость
        if np.linalg.norm(grad) < tol or np.linalg.norm(point - prev_point) < tol:
            return step, trajectory  # Возвращаем количество итераций и траекторию
    return max_steps, trajectory  # Если не сошлось, возвращаем максимум


# Методы оптимизации
def gd(point, grad, lr, state, **kwargs):
    return point - lr * grad, state


def sgd(point, grad, lr, state, **kwargs):
    noise = kwargs.get("noise", 1e-4)
    grad_noisy = grad + np.random.normal(scale=noise, size=grad.shape)
    return point - lr * grad_noisy, state


def momentum(point, grad, lr, state, **kwargs):
    momentum = kwargs.get("momentum", 0.9)
    velocity = state.get("velocity", np.zeros_like(point))
    velocity = momentum * velocity - lr * grad
    point = point + velocity
    return point, {"velocity": velocity}


def nesterov(point, grad, lr, state, **kwargs):
    momentum = kwargs.get("momentum", 0.9)
    velocity = state.get("velocity", np.zeros_like(point))
    lookahead = point + momentum * velocity
    grad = rastrigin_grad(*lookahead)
    velocity = momentum * velocity - lr * grad
    point = point + velocity
    return point, {"velocity": velocity}


def rmsprop(point, grad, lr, state, **kwargs):
    decay = kwargs.get("decay", 0.9)
    epsilon = kwargs.get("epsilon", 1e-6)
    avg_sq_grad = state.get("avg_sq_grad", np.zeros_like(point))
    avg_sq_grad = decay * avg_sq_grad + (1 - decay) * grad ** 2
    point = point - lr * grad / (np.sqrt(avg_sq_grad) + epsilon)
    return point, {"avg_sq_grad": avg_sq_grad}


def adadelta(point, grad, lr, state, **kwargs):
    decay = kwargs.get("decay", 0.95)
    epsilon = kwargs.get("epsilon", 1e-6)
    avg_sq_grad = state.get("avg_sq_grad", np.zeros_like(point))
    avg_sq_update = state.get("avg_sq_update", np.zeros_like(point))
    avg_sq_grad = decay * avg_sq_grad + (1 - decay) * grad ** 2
    update = -np.sqrt(avg_sq_update + epsilon) * grad / (np.sqrt(avg_sq_grad) + epsilon)
    avg_sq_update = decay * avg_sq_update + (1 - decay) * update ** 2
    point = point + update
    return point, {"avg_sq_grad": avg_sq_grad, "avg_sq_update": avg_sq_update}


def adam(point, grad, lr, state, **kwargs):
    beta1 = kwargs.get("beta1", 0.9)
    beta2 = kwargs.get("beta2", 0.999)
    epsilon = kwargs.get("epsilon", 1e-8)
    m = state.get("m", np.zeros_like(point))
    v = state.get("v", np.zeros_like(point))
    t = state.get("t", 0) + 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    point = point - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return point, {"m": m, "v": v, "t": t}


def nadam(point, grad, lr, state, **kwargs):
    beta1 = kwargs.get("beta1", 0.9)
    beta2 = kwargs.get("beta2", 0.999)
    epsilon = kwargs.get("epsilon", 1e-8)
    m = state.get("m", np.zeros_like(point))
    v = state.get("v", np.zeros_like(point))
    t = state.get("t", 0) + 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    m_nadam = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1 ** t)
    point = point - lr * m_nadam / (np.sqrt(v_hat) + epsilon)
    return point, {"m": m, "v": v, "t": t}

    # Настройки оптимизации
methods = {
    "GD": gd,
    "SGD": sgd,
    "Momentum": momentum,
    "Nesterov": nesterov,
    "RMSProp": rmsprop,
    "AdaDelta": adadelta,
    "Adam": adam,
    "Nadam": nadam,
}
init_point = [5, 5]
lr_dict = {
    "GD": 0.001,
    "SGD": 0.001,
    "Momentum": 0.001,
    "Nesterov": 0.001,
    "RMSProp": 0.001,
    "AdaDelta": 1.0,
    "Adam": 0.001,
    "Nadam": 0.001,
}

# Сравнение количества итераций
results = {}
trajectories = {}
for name, method in methods.items():
    lr = lr_dict[name]
    iterations, trajectory = optimize_convergence(method, rastrigin_grad, lr, init_point, tol=1e-6,
                                                  max_steps=10000, log_trajectory=True, momentum=0.9, beta1=0.9,
                                                  beta2=0.999)
    results[name] = iterations
    trajectories[name] = trajectory

# Вывод результатов
print("Количество итераций до сходимости для каждого метода:")
for method, iterations in results.items():
    print(f"{method}: {iterations} итераций")

# Визуализация траекторий


