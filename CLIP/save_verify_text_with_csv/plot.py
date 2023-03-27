import matplotlib.pyplot as plt
import numpy as np


poison_only_all = np.load('poison_only_all.npz')['arr_0']
poison_only_target = np.load('poison_only_target.npz')['arr_0']

x = np.arange(poison_only_all.shape[0])

plt.figure()
plt.plot(x, poison_only_all.mean(axis=1), label='all')
plt.plot(x, poison_only_target.mean(axis=1), label='target')

plt.legend()
plt.savefig('poison_only.png')