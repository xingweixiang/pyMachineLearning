#逻辑回归
# -*-coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

z = np.arange(-10,10,0.1)
p = sigmoid(z)
plt.plot(z,p)
#画一条竖直线，如果不设定x的值，则默认是0
plt.axvline(x=0, color='k')
plt.axhspan(0.0, 1.0,facecolor='0.7',alpha=0.4)
# 画一条水平线，如果不设定y的值，则默认是0
plt.axhline(y=1, ls='dotted', color='0.4')
plt.axhline(y=0, ls='dotted', color='0.4')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.ylim(-0.1,1.1)
#确定y轴的坐标
plt.yticks([0.0, 0.5, 1.0])
plt.ylabel('$\phi (z)$')
plt.xlabel('z')
ax = plt.gca()
ax.grid(True)
plt.show()