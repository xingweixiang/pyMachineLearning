from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y < 2, :2]
y = y[y < 2]
# 数据归一化（SVC涉及距离，应该使用数据归一化处理）
from sklearn.preprocessing import StandardScaler

stdScaler = StandardScaler()
stdScaler.fit(X)
X_standard = stdScaler.transform(X)
# 实例化svc对象，训练模型
from sklearn.svm import LinearSVC

svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)


def plot_svc_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100))
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)

    # 除去决策边界外，还要画出svc支撑向量的线
    w = model.coef_[0]
    b = model.intercept_[0]

    # w0x0 + w1x1 + b = 0
    # => x1 = -w0/w1*w0-b/w1

    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]

    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')


plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1])
plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1])
plt.show()