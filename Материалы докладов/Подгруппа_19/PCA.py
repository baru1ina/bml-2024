import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import copy

def idx_to_choose(eigvals, type_of_throw, p):
    """Eigvals - это дисперсия в новом базисе, а так как задача - понизить
    размерность, то понижаем до определённой размерности или по permitted percentage"""
    unique_sorted = sorted(eigvals)
    sum = np.sum(eigvals)

    loss = 0
    if type_of_throw == "by permitted percentage":
        sum_to_throw = sum*p/100
        i = 0

        while(loss <= sum_to_throw):
            loss += unique_sorted[i]
            i += 1

        mins = unique_sorted[:(i-1)]
        print("lost components", i - 1)
    elif type_of_throw == "by number of objects to throw up":
        mins = unique_sorted[:p]
        for i in range(p):
            loss += mins[i]
        loss = loss/sum*100
        print("lost percents of variance", loss)
        
    else:
        mins = []
        print("no loss")

    indices = [i for i in range(len(eigvals)) if eigvals[i] not in mins]
    
    return indices, loss

def PCA(data, type_of_throw = "None", p = 0):
    # centring the sample
    cop_data = copy.deepcopy(data)
    means = []
    stds = []
    # Стандартизация
    for i in range(len(cop_data)):
        mean = np.mean(cop_data[i])        
        std = np.std(cop_data[i])
        means.append(mean)
        stds.append(std)
        for j in range(len(cop_data[i])):
            cop_data[i][j] = (cop_data[i][j] - mean)/std

    # cov matrix and fing eigen -vectors/-values
    covmat = np.cov(cop_data)
    eigvals, eigvecs = np.linalg.eig(covmat)

    # eigval = var of hidden component, so we can throw up unnecessary ones
    indices, loss = idx_to_choose(eigvals, type_of_throw, p)
    

    # projections is (V^T*X), V - matrix of eigvecs
    V = []
    for idx in indices:
        V.append(eigvecs[:, idx])
    new_data = np.dot(V, cop_data)

    # recover the real values:
    rec_data = np.dot(np.transpose(V), new_data)
    for i in range(len(rec_data)):
        for j in range(len(rec_data[i])):
            rec_data[i][j] = rec_data[i][j] * stds[i]
            rec_data[i][j] += means[i]


# print data
    # print("data")
    # print(np.array(data))
    # print()

    # # for print median data (with less dimension)
    # print("med data")
    # print(new_data)
    # print()

    # print("recovered data")
    # print(rec_data)
    # print()

    return rec_data, loss

def main():

    # test 1
    # n = 10
    # x1 = [i for i in range(n)]
    # x2 = [(-2)*i for i in range(n)]
    # x3 = np.random.normal(0, 5, size = n)
    # x4 = np.random.normal(0, 10, size = n)
    # data = [x1, x2, x3, x4]

    #test 2: visualisation
    n = 50
    x1 = [i for i in range(n)]
    s1 =np.random.normal(0, 5, size = n)
    y1 = x1 + s1

    x2 = [(2*i + 5) for i in range(n)]
    s2 =np.random.normal(0, 5, size = n)
    y2 = x2 + s2

    x3 = [((-3)*i + 10) for i in range(n)]
    s3 = np.random.normal(0, 5, size = n)
    y3 = s3

    data = [y1, y2, y3]

    # options: "by permitted percentage"(enter percents to throw) and "by number of objects to throw up"(enter the number to throw)
    rec_data, loss = PCA(data, "by number of objects to throw up", 1)
    
    # plotting
    x1 = data[0]
    y1 = data[1]
    z1 = data[2]    
    x2 = rec_data[0]
    y2 = rec_data[1]
    z2 = rec_data[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, color='green', label='Первоначальные данные - 3-мерная модель')
    ax.scatter(x2, y2, z2, color='blue', label='Проекция - 2-мерная модель')
    
    ax.set_title('Пример: снижение размерности 3-мерной модели до 2-мерной.')
    ax.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()