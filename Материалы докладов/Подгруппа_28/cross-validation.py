import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class KNearestNeighbors:

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dist = self.compute_distances(X)
        return self.predict_labels(dist, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]), axis=1))
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            idx = np.argsort(dists[i,:])[:k]
            neighbors = self.y_train[idx].ravel()
            cntr = Counter(neighbors)
            y_pred[i] = cntr.most_common(1)[0][0]
        return y_pred

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Training data shape', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape', X_test.shape)
print('Test labels shape', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    indxs = np.flatnonzero(y_train==y)
    indxs = np.random.choice(indxs, samples_per_class, replace=False)
    for i, idx in enumerate(indxs):
        plt_idx = i* num_classes+y+1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()

num_training = 1000
indx = range(num_training)
X_train = X_train[indx].astype('float')
y_train = y_train[indx].astype('int')

num_test = 10
indx = range(num_test)
X_test = X_test[indx].astype('float')
y_test = y_test[indx].astype('int')

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

num_folds = 5
neighbores = {1, 5, 7, 15, 20}

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_acc = {}

for k in neighbores:
    k_to_acc[k] = list()

for k in neighbores:
    print('k =', k)
    for i in range(num_folds):
        X_val = X_train_folds[i]
        y_val = y_train_folds[i]
        X_tr = np.vstack((X_train_folds[0:i]+X_train_folds[i+1:]))
        y_tr = np.vstack((y_train_folds[0:i]+y_train_folds[i+1:])).ravel()

        knn=KNearestNeighbors()
        knn.train(X_tr, y_tr)
        pred = knn.predict(X_val, k = k)

        acc = np.mean(pred==y_val)
        k_to_acc[k].append(acc)


print('FINISHED')

best_k = 0
best_acc = 0
for k in neighbores:
    acc = np.mean(k_to_acc[k])
    if acc > best_acc:
        best_acc = acc
        best_k = k

clf = KNearestNeighbors()
clf.train(X_train, y_train)
y_test_pred = clf.predict(X_test, k=best_k)

acc = np.mean(y_test_pred==y_test)

print('Acc:{}, best_k:{}'.format(acc, best_k))


        


