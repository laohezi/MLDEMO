from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
print("data:", data)
features = data['data']

feature_names = data['feature_names']
target = data['target']
labels = data['target_names']

# for t, marker, c in zip(range(3), ">ox", "rgb"):
#     plt.scatter(features[target == t, 0],
#                 features[target == t, 1],
#                 marker=marker,
#                 c=c)
#     plt.show()

plength = features[:, 2]
print("features:", features)
print("plength:", plength)
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa:{0}.'.format(max_setosa))
print('Minimum of setosa:{0}.'.format(min_non_setosa))

if features[:, 2].any() < 2: print('Iris Setosa')

features = features[~is_setosa]
labels = labels[~is_setosa]

virginica = (labels == 'virginica')

best_acc = -1.0
print(features.shape[1])
for fi in range(features.shape[1]):
    print("fi:", fi)
    thresh = features[:fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == virginica).mean()

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
    print("best_acc:", best_acc)
    print("best_fi:", best_fi)
    print("best_t", best_t)
