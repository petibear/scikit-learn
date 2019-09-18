"""
==========================================
IsolationForest `max_depth` example
==========================================

Comparison of :class:`sklearn.ensemble.IsolationForest` runs with
various (iTrees) depth setting.


"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# generate train data
X = make_blobs([500, 20, 20, 10, 15],
               n_features=2,
               centers=[[0, 0], [1.3, 1.3], [-1, -1], [-1, 2], [1, -2]],
               cluster_std=[0.4, 0.1, 0.1, 0.05, 0.05],
               random_state=rng)
X_train = X[0]

# define parameters
params_shared = {'random_state': rng, 'contamination': 0.1, 'n_estimators': 20}
runs = [
    {'title': 'A - Current',
     'params': {'max_samples': 16}},
    {'title': 'B - Current - Small sample',
     'params': {'max_samples': 2}},
    {'title': 'C - New - Fully Grown',
     'params': {'max_samples': 16, 'max_depth': "max"}},
    {'title': 'D - New - Limited at 2',
     'params': {'max_samples': 16, 'max_depth': 1}}
]

# fit and plot
clfs = []
levels = np.linspace(0.2, 0.8, 13)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
plt.suptitle('iForest - maximum tree depths comparison (n_estimators = {})'
             .format(params_shared['n_estimators']),
             fontsize='xx-large', weight='bold')

for idx, r in enumerate(runs, 1):
    params = r['params']
    params = dict([(k, v) for (k, v) in params_shared.items() if k not in params] + list(params.items()))
    clf = IsolationForest(**params)
    clf.fit(X_train)
    y_pred = clf.predict(X_train)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = -clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(2, 2, idx)
    act_max = max([et.get_depth() for et in clf.estimators_])
    title = '{} | {}'.format(r['title'],
                             ', '.join(['{}={}'.format(k, v)
                                        for k, v in r['params'].items()]))
    stitle = '(actual max_depth={})'.format(act_max)
    plt.title("{}\n{}".format(title, stitle))
    ax = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r,levels=levels,
                      vmin=0, vmax=1)
    levels = ax.levels
    plt.scatter(x=X_train[:, 0], y=X_train[:, 1],
                c='white', s=40, edgecolor='k')
    clfs.append(clf)
fig.subplots_adjust(right=0.77)
cbar = fig.colorbar(mappable=ax, ax=axes.ravel().tolist(), fraction=0.15)
cbar.set_label('Anomaly Score')
plt.show()
